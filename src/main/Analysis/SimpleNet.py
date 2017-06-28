#should refactor out common code
from SimpleForest import forest_training_data
import _constants


import tensorflow as tf
import pandas as pd


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def select_batch(tensor, index, size):
    return tensor

def build_net(train, winloss):
    inputs = 47
    outputs = 1
    batch_size = 64
    layer_widths = [100, 50]
    eps = 1e-3
    print("TF WIZARDRY")
    x = train #= tf.placeholder(tf.float32, shape=[None, inputs])
    resp = winloss #tf.placeholder(tf.float32, shape=[None, outputs])
    print("Next: layers")
    weights = [weight_variable([inputs, layer_widths[0]])]
    biases = [bias_variable([layer_widths[0]])]
    for i in range(1, len(layer_widths)):
        weights.append(weight_variable([layer_widths[i - 1], layer_widths[i]]))
        biases.append(bias_variable([layer_widths[i]]))
    final_layer = weight_variable([layer_widths[-1], outputs])
    final_bias = bias_variable([outputs])
    layer_outputs = [tf.nn.elu(tf.matmul(x, weights[0]) + biases[0])]
    for i in range(1, len(layer_widths)):
        layer_outputs.append(tf.nn.elu(tf.matmul(layer_outputs[i - 1], weights[i]) + biases[i]))
    out = tf.matmul(layer_outputs[-1], final_layer) + final_bias
    out = tf.nn.sigmoid(out)
    print("Next: loss")
    loss = -tf.reduce_mean(tf.log(resp * out + (1 - resp) * (1 - out) + eps))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct = tf.equal(resp, tf.round(out))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    print("Hi")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            #batch = [select_batch(train, i, batch_size), select_batch(winloss, i, batch_size)];
            train_step.run()#feed_dict={x: batch[0], resp: batch[1]})
            if i % 100 == 0:
                #print(tf.slice(winloss, [0], [50]).eval())
                #print(tf.slice(train, [0, 0], [5, 5]).eval())
                print("acc:" + str(accuracy.eval()))#feed_dict={x: batch[0], resp: batch[1]}))
                print("loss:" + str(loss.eval()))



def train_net(df):
    pred, resp = forest_training_data(df)
    print("Fetched data, tensoring")
    tpred = tf.cast(tf.stack(pred), tf.float32)
    print("Half tensored")
    tresp = tf.cast(tf.stack(resp), tf.float32)
    tresp = tf.reshape(tresp, [972, 1])
    build_net(tpred, tresp)


df = pd.read_csv(_constants.data_location + 'simple_game_data_leagueId={}.csv'.format(3))

train_net(df)











