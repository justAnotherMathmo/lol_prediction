import tensorflow as tf
import numpy as np
import edward as ed
import Nets
import itertools
import SimpleForest
import _constants
import pandas as pd
import sklearn as sk

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

    winloss = list(df.win)
    for i, w in zip(range(len(results)), winloss):
        results[i][-1] = w

    return games, results


class BMixture(ed.RandomVariable):

    def __init__(self, ber, d1, d2):
        self.ber = ber
        self.d1 = d1
        self.d2 = d2

    def log_prob(self, x):
         


class FactoredPredictor:

    def __init__(self, nn):
        self.nn = nn

    def apply(self, x, params):
        y = self.nn.apply(x, params[:-3])



    def params(self):
        return self.nn.params + []

class PairModel(object):

    def __init__(self, teams, predictor, features=8, prior=None):

        self.team_numbers = {}
        self.team_names = []
        self.team_vectors = Nets.gauss_var_post([teams, features])
        self.predictor = predictor
        self.param_space = list(predictor.params()) + [self.team_vectors.get_shape()]
        self.var_post = [Nets.guass_var_post(shape) for shape in self.param_space]
        if prior is None:
            prior = [Nets.gauss_prior(shape) for shape in self.param_space]
        self.prior = prior

    def train_model(self, games, results, num_train_steps=10000):
        params_post = {p: q for p, q in zip(self.prior, self.var_post)}
        x = tf.placeholder(tf.int32, shape=[None, 3])
        print('accuracy, log_likelihood, crossentropy', ed.evaluate(['accuracy', 'log_likelihood', 'crossentropy'], data={results: results, x: games}))
        y = self.predict(x)

        inference = ed.KLqp(self.var_post, data={y: results, x: games})

        inference.run(n_samples=16, n_iter=num_train_steps)

        # Get output object dependant on variational posteriors rather than priors
        out_post = ed.copy(y, params_post)
        # Re-evaluate metrics
        print('accuracy, log_likelihood, crossentropy',
              ed.evaluate(['accuracy', 'log_likelihood', 'crossentropy'], data={out_post: results, x: games}))

    def predict(self, x):
        return self.predictor.apply(tf.concat(tf.gather(self.team_vectors, x[:-1]), x[-1:]))


def read_csv(league):
    return pd.read_csv(_constants.data_location + 'simple_game_data_leagueId={}.csv'.format(league))

games, results = read_data(read_csv(2))
print(results)
