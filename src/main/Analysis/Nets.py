import tensorflow as tf
import edward as ed
import numpy as np
import itertools


# Gaussian variational 'posterior' with tf.Variable parameters - to be fit with the true posterior
def gauss_var_post(shape):
    return ed.models.Normal(loc=tf.Variable(tf.zeros(shape)), scale=tf.nn.softplus(tf.Variable(tf.zeros(shape))))


# guassian prior (with requested shape)
def gauss_prior(shape, std=1.0):
    return ed.models.Normal(loc=tf.zeros(shape), scale=std*tf.ones(shape))


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


def merge(a, b):
    for i, j in zip(a, b):
        yield i
        yield j


class SuperDenseNet(object):

    def __init__(self, inputs, layer_widths, activations, lookback=None):
        if lookback is None:
            lookback = len(layer_widths)
        self.lookback = lookback
        self.layer_widths = layer_widths
        self.inputs = inputs
        widths = [inputs] + layer_widths[:-1]
        self.input_widths = [sum(widths[max(0, i - lookback):i + 1]) for i in range(len(widths))]
        self.activations = activations
        self.tensor_shapes = [[low, high]
                              for low, high in zip(self.input_widths, layer_widths)]

    def weight_shapes(self):
        return self.tensor_shapes

    def bias_shapes(self):
        return self.layer_widths

    def param_space(self):
        return self.weight_shapes() + self.bias_shapes()

    """
    def apply(self, x, weights, biases):
        layer_outputs = [x]
        for a, w, b in zip(self.activations, weights, biases):
            layer_outputs.append(a(tf.matmul(tf.concat(layer_outputs[max(-self.lookback, -len(layer_outputs)):], 1), w) + b))
        return layer_outputs[-1]
    """

    def outputs(self):
        return self.layer_widths[-1]

    def apply(self, x, params):
        layer_outputs = [x]
        weights = params[:len(self.weight_shapes())]
        biases = params[len(self.weight_shapes()):]
        for a, w, b in zip(self.activations, weights, biases):
            layer_outputs.append(
                a(tf.matmul(tf.concat(layer_outputs[max(-self.lookback, -len(layer_outputs)):], 1), w) + b))
        return layer_outputs[-1]

