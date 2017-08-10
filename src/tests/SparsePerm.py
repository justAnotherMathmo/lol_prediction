import tensorflow as tf
import edward as ed
import numpy as np


class PermLayer(object):

    def __init__(self, perm):
        self.perm = perm

    def apply(self, x):
        return tf.gather_nd(x, tf.stack([self.perm]))

class SparseBlockLayer(object):

    def __init__(self, neurons, positions):
        self.neurons = neurons
        self.positions = positions
    def apply(self, x):
