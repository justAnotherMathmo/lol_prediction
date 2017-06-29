# should refactor out common code
from SimpleForest import forest_training_data
from SimpleForest import predict_league
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


def build_net(train, winloss, num_train_steps=10000):
    inputs = 47
    outputs = 1
    # batch_size = 64
    layer_widths = [128, 128, 64, 32]
    eps = 1e-3
    # x = train  # = tf.placeholder(tf.float32, shape=[None, inputs])
    resp = winloss  # tf.placeholder(tf.float32, shape=[None, outputs])

    weights = [weight_variable([low, high])
               for low, high in zip([inputs]+layer_widths, layer_widths + [outputs])]
    biases = [bias_variable([width]) for width in layer_widths + [outputs]]

    out = train
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        layer_outputs = tf.matmul(out, weight) + bias
        if index < len(weights)-1:
            out = tf.nn.elu(layer_outputs)
        else:
            out = tf.nn.sigmoid(layer_outputs)

    loss = -tf.reduce_mean(tf.log(resp * out + (1 - resp) * (1 - out) + eps))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct = tf.equal(resp, tf.round(out))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(num_train_steps):
            # batch = [select_batch(train, i, batch_size), select_batch(winloss, i, batch_size)];
            train_step.run()  # feed_dict={x: batch[0], resp: batch[1]})

        print("acc:" + str(accuracy.eval()))  # feed_dict={x: batch[0], resp: batch[1]}))
        print("loss:" + str(loss.eval()))


def train_net(df):
    pred, resp = forest_training_data(df)
    # print("Fetched data, tensoring")
    tpred = tf.cast(tf.stack(pred), tf.float32)

    tpred = tf.nn.l2_normalize(tpred, 0)
    # print("Half tensored")
    tresp = tf.cast(tf.stack(resp), tf.float32)
    tresp = tf.reshape(tresp, [len(resp), 1])
    build_net(tpred, tresp)


if __name__ == '__main__':
    league = 3
    df = pd.read_csv(_constants.data_location + 'simple_game_data_leagueId={}.csv'.format(league))
    train_net(df)
    predict_league(league)











