import tensorflow as tf
import numpy as np
import edward as ed
import Nets
import itertools
import SimpleForest

import _constants
import pandas as pd
import sklearn as sk
from tensorflow.contrib.distributions import NOT_REPARAMETERIZED


def read_data(df):
    simple_df = SimpleForest.simplify_dataframe(df)
    team_numbers = {}
    teams_seen = 0
    games = []
    results = []

    def reg_team(name):
        if name in team_numbers:
            return team_numbers[name]
        nonlocal teams_seen
        team_numbers[name] = teams_seen
        teams_seen = teams_seen + 1
        return teams_seen - 1

    for row_num in range(len(simple_df) // 2):
        blue = simple_df.ix[2 * row_num]
        red = simple_df.ix[2 * row_num + 1]
        blueN = reg_team(blue['team_name'])
        redN = reg_team(red['team_name'])
        games.append(np.array([blueN, redN, 0]))
        games.append(np.array([redN, blueN, 1]))
        #Need to give from both rows to both entries!
        blueStat = simple_df.drop('team_name', axis=1).ix[2 * row_num].as_matrix()
        redStat = simple_df.drop('team_name', axis=1).ix[2 * row_num + 1].as_matrix()
        blueStat = blueStat.astype(np.float32)
        redStat = redStat.astype(np.float32)
        blueStat[np.isnan(blueStat)] = 0
        redStat[np.isnan(redStat)] = 0
        results.append(np.append(np.append(blueStat, redStat[:2]), redStat[4:11]))
        results.append(np.append(np.append(redStat, blueStat[:2]), blueStat[4:11]))
    print(results[0])
    results = sk.preprocessing.scale(results, copy=False)
    results = results.astype(np.float32)
    winloss = list(df.win)
    for i, w in zip(range(len(results)), winloss):
        results[i][-1] = w

    return games, results, teams_seen


class IndependentJoint(ed.RandomVariable, tf.contrib.distributions.Distribution):

    def __init__(self, params,  # d1, d2, d1dim,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="IndependentJoint", *args, **kwargs):

        d1, d2, d1dim = params

        parameters = locals()
        parameters.pop("self")
        with tf.name_scope(name, values=[params]):
            with tf.control_dependencies([]):
                self.d1 = d1
                self.d2 = d2
                self.d1dim = d1dim

        super(IndependentJoint, self).__init__(
            dtype=tf.float32,
            reparameterization_type=NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self.d1.value(), self.d2.value()],
            name=name,
            *args,
            **kwargs)

    def _log_prob(self, x):
        return self.d1.log_prob(x[:, :self.d1dim]) + self.d2.log_prob(x[:, self.d1dim:])

    def _sample_n(self, n, seed=None):
        return tf.concat(self.d1._sample_n(n, seed), self.d2._sample_n(n, seed), 1)


class FactoredPredictor:

    def __init__(self, nn, outputs):
        self.nn = nn
        self.outputs = outputs

    def apply(self, x, params):
        y = self.nn.apply(x, params[:-2])
        return IndependentJoint((ed.models.MultivariateNormalTriL(tf.matmul(y, params[-3]), tf.tensordot(y, params[-2], 1)),
                                ed.models.Bernoulli(logits=tf.matmul(y, params[-1])), 31),
                                value=tf.zeros([1320, 32]))

    def param_space(self):
        return self.nn.param_space() + [[self.nn.outputs(), self.outputs - 1],
                                        [self.nn.outputs(), self.outputs - 1, self.outputs - 1],
                                        [self.nn.outputs(), 1]
                                        ]

class PairModel(object):

    def __init__(self, teams, predictor, features=8, prior=None):

        self.team_numbers = {}
        self.team_names = []
        self.features = features
        self.predictor = predictor
        self.param_space = list(predictor.param_space()) + [[teams, features]]
        self.var_post = [Nets.gauss_var_post(shape) for shape in self.param_space]
        if prior is None:
            prior = [Nets.gauss_prior(shape) for shape in self.param_space]
        self.prior = prior

    def train_model(self, games, results, num_train_steps=10000):
        params_post = {p: q for p, q in zip(self.prior, self.var_post)}
        x = tf.placeholder(tf.int32, shape=[None, 3])

        y = self.predict(x)
        print('accuracy, log_likelihood',
              ed.evaluate(['accuracy', 'log_likelihood'], data={y: results, x: games}))
        inference = ed.KLqp(params_post, data={y: results, x: games})

        inference.run(n_samples=16, n_iter=num_train_steps)

        # Get output object dependant on variational posteriors rather than priors
        out_post = ed.copy(y, params_post)
        # Re-evaluate metrics
        print('accuracy, log_likelihood',
              ed.evaluate(['accuracy', 'log_likelihood'], data={out_post: results, x: games}))

    def predict(self, x):
        team_vectors = tf.reshape(tf.gather(self.prior[-1], x[:, :-1]), [-1, 2*self.features])
        return self.predictor.apply(tf.concat([team_vectors, tf.cast(x[:, -1:], tf.float32)], 1)
                                    , self.prior[:-1])


def read_csv(league):
    return pd.read_csv(_constants.data_location + 'simple_game_data_leagueId={}.csv'.format(league))

if __name__ == '__main__':
    games, results, teams = read_data(read_csv(2))
    print(games[0])
    outputs = 32
    print(results)
    layer_widths = [8, 8, 8, 8, 8]
    activations = [Nets.selu for _ in layer_widths] + [tf.identity]
    layer_widths += [outputs]
    net = Nets.SuperDenseNet(17, layer_widths, activations)
    predictor = FactoredPredictor(net, len(results[0]))
    myModel = PairModel(teams, predictor)

    # tf.global_variables_initializer()

    myModel.train_model(games, results)
