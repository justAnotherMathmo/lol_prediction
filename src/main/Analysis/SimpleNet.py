# should refactor out common code
from SimpleForest import forest_training_data, predict_league
import _constants
import tensorflow as tf
import pandas as pd
import edward as ed
import numpy as np
import Nets

# Gaussian variational 'posterior' with tf.Variable parameters - to be fit with the true posterior
def gauss_var_post(shape):
    return ed.models.Normal(loc=tf.Variable(tf.random_normal(shape)), scale=tf.nn.softplus(tf.Variable(tf.random_normal(shape))))


# guassian prior (with requested shape)
def gauss_prior(shape):
    return ed.models.Normal(loc=tf.zeros(shape), scale=tf.ones(shape))


# TODO
def select_batch(tensor, index, size):
    return tensor


# Builds and trains neural network given stats train, outcomes winloss
def build_net(x_train, y_train, num_train_steps=10000):
    # Number of stats currently used to predict outcome- 23 per team + variable for side
    inputs = 47
    outputs = 1

    # widths of fully-connected layers in NN
    layer_widths = [8, 8]
    # Input data goes here (via feed_dict or equiv)
    x = tf.placeholder(tf.float32, shape=[len(x_train), inputs])

    activations = [tf.nn.elu for _ in layer_widths] + [tf.identity]
    layer_widths += [outputs]
    net = Nets.SuperDenseNet(inputs, layer_widths, activations)
    # Construct all parameters of NN, set to independant gaussian priors
    weights = [gauss_prior(shape) for shape in net.weight_shapes()]
    biases = [gauss_prior(shape) for shape in net.bias_shapes()]
    out = ed.models.Bernoulli(logits=net.apply(x, weights, biases))

    # Variational 'posterior's for NN params
    qweights = [gauss_var_post(w.shape) for w in weights]
    qbiases = [gauss_var_post(b.shape) for b in biases]

    # Map from random variables to their variational posterior objects
    weights_post = {weights[i]: qweights[i] for i in range(len(weights))}
    biases_post = {biases[i]: qbiases[i] for i in range(len(weights))}
    var_post = {**weights_post, **biases_post}

    # evaluate 'accuracy' (what even is this??) and likelihood of model over the dataset before training
    print('accuracy, log_likelihood, crossentropy',
          ed.evaluate(['accuracy', 'log_likelihood', 'crossentropy'], data={out: y_train, x: x_train}))
    # Run variational inference, minimizing KL(q, p) using stochastic gradient descent over variational params
    inference = ed.KLqp(var_post, data={out: y_train, x: x_train})
    inference.run(n_samples=16, n_iter=10000)

    # Get output object dependant on variational posteriors rather than priors
    out_post = ed.copy(out, var_post)
    # Re-evaluate metrics
    print('accuracy, log_likelihood, crossentropy',
          ed.evaluate(['accuracy', 'log_likelihood', 'crossentropy'], data={out_post: y_train, x: x_train}))


def train_net(df):
    pred, resp = forest_training_data(df)
    for x in pred:
        print(x.tolist())

    apred = np.array(pred, dtype=np.float32)
    aresp = np.array(resp, dtype=np.float32).reshape((len(pred), 1))
    build_net(apred, aresp)

   # print("Fetched data, tensoring")
   #  tpred = tf.cast(tf.stack(pred), tf.float32)
   #
   #  tpred = tf.nn.l2_normalize(tpred, 0)
   #  #print("Half tensored")
   #  tresp = tf.cast(tf.stack(resp), tf.float32)
   #  tresp = tf.reshape(tresp, [len(resp), 1])
   #  build_net(tpred, tresp)


if __name__ == '__main__':
    league = 3
    df = pd.read_csv(_constants.data_location + 'simple_game_data_leagueId={}.csv'.format(league))
    train_net(df)
    predict_league(league)











