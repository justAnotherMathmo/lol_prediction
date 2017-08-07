import tensorflow as tf
import numpy as np
import edward as ed
import Nets


def read_data():
    pass


class PairModel(object):

    def __init__(self, teams, predictor, prior=None):

        self.team_numbers = {}
        self.team_names = []
        self.team_vectors = Nets.gauss_var_post([teams, ])
        self.predictor = predictor
        self.var_post = [Nets.guass_var_post(shape) for shape in predictor.params]
        if prior is None:
            prior = [Nets.gauss_prior(shape) for shape in predictor.params()]
        self.prior = prior

    def train_model(self, games, results, num_train_steps=10000):
        params_post = {p: q for p, q in zip(self.prior, self.var_post)}
        x = tf.placeholder(tf.int32, shape=[None, 2])
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
        return self.predictor.apply(tf.gather(self.team_vectors, x))

