# should refactor out common code
from SimpleForest import forest_training_data, predict_league
import _constants
import tensorflow as tf
import pandas as pd
import edward as ed
import numpy as np


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
def build_net(train, winloss, num_train_steps=10000):
    # Number of stats currently used to predict outcome- 23 per team + variable for side
    inputs = 47
    outputs = 1

    # widths of fully-connected layers in NN
    layer_widths = [128, 64, 32]
    # Input data goes here (via feed_dict or equiv)
    x = tf.placeholder(tf.float32, shape=[len(train), inputs])
    resp = winloss

    # Construct all parameters of NN, set to independant gaussian priors
    weights = [gauss_prior([low, high])
               for low, high in zip([inputs] + layer_widths, layer_widths + [outputs])]
    biases = [gauss_prior([width]) for width in layer_widths + [outputs]]

    # Define operation of neural network on input
    # TODO fix nicely - I don't like this
    layer_outputs = [tf.nn.elu(tf.matmul(tf.nn.l2_normalize(x, 0), weights[0]) + biases[0])]
    for i in range(1, len(layer_widths)):
        layer_outputs.append(tf.nn.elu(tf.matmul(layer_outputs[i - 1], weights[i]) + biases[i]))
    # Define output distribution
    out = ed.models.Bernoulli(logits=tf.matmul(layer_outputs[-1], weights[-1]) + biases[-1])

    # Variational 'posterior's for NN params
    qweights = [gauss_var_post(w.shape) for w in weights]
    qbiases = [gauss_var_post(b.shape) for b in biases]

    # Map from random variables to their variational posterior objects
    weights_post = {weights[i]: qweights[i] for i in range(len(weights))}
    biases_post = {biases[i]: qbiases[i] for i in range(len(weights))}
    var_post = {**weights_post, **biases_post}

    # evaluate 'accuracy' (what even is this??) and likelihood of model over the dataset before training
    print('binary_accuracy', ed.evaluate('binary_accuracy', data={out: resp, x: train}))
    print('log_likelihood', ed.evaluate('log_likelihood', data={out: resp, x: train}))
    # Run variational inference, minimizing KL(q, p) using stochastic gradient descent over variational params
    inference = ed.KLqp(var_post, data={out: resp, x: train})
    inference.run(n_samples=8, n_iter=1000)

    # Get output object dependant on variational posteriors rather than priors
    out_post = ed.copy(out, var_post)
    # Re-evaluate metrics
    print('binary_accuracy', ed.evaluate('binary_accuracy', data={out_post: resp, x: train}))
    print('log_likelihood', ed.evaluate('log_likelihood', data={out_post: resp, x: train}))

    # Run our own metrics because we don't know what the hell evaluate('binary_accuracy' is doing)
    correct = tf.equal(resp, tf.round(out_post.probs))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print(correct.get_shape())
        print(accuracy.eval(feed_dict={x: train}))
        # print(qfinal_layer.mean().eval(feed_dict={x: train}))
        # print(qfinal_layer.scale.eval(feed_dict={x: train}))
        # print(out_post.probs.eval(feed_dict={x: train}))
    # print("Next: loss")
    # loss = -tf.reduce_mean(tf.log(resp * out + (1 - resp) * (1 - out) + eps))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # correct = tf.equal(resp, tf.round(out))
    # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    # #print("Hi")
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for i in range(10000):
    #         #batch = [select_batch(train, i, batch_size), select_batch(winloss, i, batch_size)];
    #         train_step.run()#feed_dict={x: batch[0], resp: batch[1]})
    #
    #     print("acc:" + str(accuracy.eval()))#feed_dict={x: batch[0], resp: batch[1]}))
    #     print("loss:" + str(loss.eval()))


def train_net(df):
    pred, resp = forest_training_data(df)
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











