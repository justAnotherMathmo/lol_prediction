#should refactor out common code
from SimpleForest import forest_training_data
from SimpleForest import predict_league
import _constants


import tensorflow as tf
import pandas as pd
import edward as ed
import numpy as np

def gauss_var_post(shape):
  return ed.models.Normal(loc=tf.Variable(tf.random_normal(shape)), scale=tf.nn.softplus(tf.Variable(tf.random_normal(shape))))

def gauss_prior(shape):
    return ed.models.Normal(loc=tf.zeros(shape), scale=tf.ones(shape))

def select_batch(tensor, index, size):
    return tensor

def build_net(train, winloss):
    inputs = 47
    outputs = 1
    #batch_size = 64
    layer_widths = [128, 64, 32]
    x = tf.placeholder(tf.float32, shape=[len(train), inputs])
    resp = winloss #tf.placeholder(tf.float32, shape=[None, outputs])
    weights = [gauss_prior([inputs, layer_widths[0]])]
    biases = [gauss_prior([layer_widths[0]])]
    for i in range(1, len(layer_widths)):
        weights.append(gauss_prior([layer_widths[i - 1], layer_widths[i]]))
        biases.append(gauss_prior([layer_widths[i]]))
    final_layer = gauss_prior([layer_widths[-1], outputs])
    final_bias = gauss_prior([outputs])
    layer_outputs = [tf.nn.elu(tf.matmul(tf.nn.l2_normalize(x, 0), weights[0]) + biases[0])]
    for i in range(1, len(layer_widths)):
        layer_outputs.append(tf.nn.elu(tf.matmul(layer_outputs[i - 1], weights[i]) + biases[i]))
    out = ed.models.Bernoulli(logits=tf.matmul(layer_outputs[-1], final_layer) + final_bias)

    qweights = [gauss_var_post(w.shape) for w in weights]
    qbiases = [gauss_var_post(b.shape) for b in biases]
    qfinal_layer = gauss_var_post(final_layer.shape)
    qfinal_bias = gauss_var_post(final_bias.shape)

    var_post = {final_layer: qfinal_layer, final_bias: qfinal_bias}
    for i in range(len(weights)):
        var_post[weights[i]] = qweights[i]
    for i in range(len(biases)):
        var_post[biases[i]] = qbiases[i]
    print(ed.evaluate('binary_accuracy', data={out: resp, x: train}))
    print(ed.evaluate('log_likelihood', data={out: resp, x: train}))
    inference = ed.KLqp(var_post, data={out: resp, x: train})
    inference.run(n_samples=8, n_iter=1000)
    out_post = ed.copy(out, var_post)
    print(ed.evaluate('binary_accuracy', data={out_post: resp, x: train}))
    print(ed.evaluate('log_likelihood', data={out_post: resp, x: train}))

    correct = tf.equal(resp, tf.round(out_post.probs))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(correct.get_shape())
        print(accuracy.eval(feed_dict={x: train}))
        #print(qfinal_layer.mean().eval(feed_dict={x: train}))
        #print(qfinal_layer.scale.eval(feed_dict={x: train}))
        #print(out_post.probs.eval(feed_dict={x: train}))
    #print("Next: loss")
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
    apred = np.array(pred).astype(np.float32)
    aresp = np.array(resp).astype(np.float32).reshape((len(pred), 1))
    build_net(apred, aresp)
   # print("Fetched data, tensoring")
   #  tpred = tf.cast(tf.stack(pred), tf.float32)
   #
   #  tpred = tf.nn.l2_normalize(tpred, 0)
   #  #print("Half tensored")
   #  tresp = tf.cast(tf.stack(resp), tf.float32)
   #  tresp = tf.reshape(tresp, [len(resp), 1])
   #  build_net(tpred, tresp)

league = 3
df = pd.read_csv(_constants.data_location + 'simple_game_data_leagueId={}.csv'.format(league))

train_net(df)
#predict_league(league)











