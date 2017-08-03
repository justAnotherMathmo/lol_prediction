import tensorflow as tf
import edward as ed
import numpy as np
import itertools


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))

class SuperDenseNet(object):

    def __init__(self, inputs, layer_widths, activations):
        self.layer_widths = layer_widths
        self.inputs = inputs
        self.input_widths = itertools.accumulate([inputs] + layer_widths[:-1])
        self.activations = activations
        self.tensor_shapes = [[low, high]
                              for low, high in zip(self.input_widths, layer_widths)]
    def weight_shapes(self):
        return self.tensor_shapes
    def bias_shapes(self):
        return self.layer_widths

    def apply(self, x, weights, biases):
        layer_outputs = [x]
        for a, w, b in zip(self.activations, weights, biases):
            layer_outputs.append(a(tf.matmul(tf.concat(layer_outputs, 1), w) + b))
        return layer_outputs[-1]

