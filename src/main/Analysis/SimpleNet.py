# should refactor out common code
from SimpleForest import forest_training_data, predict_league
import _constants
import tensorflow as tf
import pandas as pd
import edward as ed
import numpy as np
import sklearn as sk
import Nets
from yellowfin import YFOptimizer




# TODO
def select_batch(tensor, index, size):
    return tensor


# Builds and trains neural network given stats train, outcomes winloss
def build_net(x_train, y_train, num_train_steps=10000, x_test=None, y_test=None):
    # Number of stats currently used to predict outcome- 23 per team + variable for side
    inputs = 47
    outputs = 1
    if x_test is None:
        x_test = x_train
    if y_test is None:
        y_test = y_train
    # widths of fully-connected layers in NN

    # Input data goes here (via feed_dict or equiv)
    x = tf.placeholder(tf.float32, shape=[None, inputs])
    layer_widths = [16, 16, 16, 16, 16, 16]
    activations = [Nets.selu for _ in layer_widths] + [tf.identity]
    layer_widths += [outputs]
    net = Nets.SuperDenseNet(inputs, layer_widths, activations)
    # Construct all parameters of NN, set to independant gaussian priors
    params = [Nets.gauss_prior(shape) for shape in net.param_space()]

    out = ed.models.Bernoulli(logits=net.apply(x, params))

    # Variational 'posterior's for NN params
    qparams = [Nets.gauss_var_post(w.shape) for w in params]
    asd = tf.train.AdamOptimizer

    # Map from random variables to their variational posterior objects
    params_post = {params[i]: qparams[i] for i in range(len(params))}

    # evaluate accuracy and likelihood of model over the dataset before training
    print('accuracy, log_likelihood, crossentropy',
          ed.evaluate(['accuracy', 'log_likelihood', 'crossentropy'], data={out: y_test, x: x_test}))
    # Run variational inference, minimizing KL(q, p) using stochastic gradient descent over variational params

    inference = ed.KLqp(params_post, data={out: y_train, x: x_train})
    #inference.initialize(optimizer=YFOptimizer())

    inference.run(n_samples=32, n_iter=num_train_steps)

    # Get output object dependant on variational posteriors rather than priors
    out_post = ed.copy(out, params_post)
    # Re-evaluate metrics
    print('accuracy, log_likelihood, crossentropy',
          ed.evaluate(['accuracy', 'log_likelihood', 'crossentropy'], data={out_post: y_test, x: x_test}))



def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def train_net(df):
    pred, resp = forest_training_data(df)
    #for x in pred:
      #  print(x.tolist())
    print(len(resp))
    sk.preprocessing.scale(pred, copy=False)
    apred = np.array(pred, dtype=np.float32)
    aresp = np.array(resp, dtype=np.float32).reshape((len(pred), 1))
    shuffle_in_unison(apred, aresp)
    val_set = 200
    x_train = apred[0:-val_set]
    y_train = aresp[0:-val_set]
    x_test = apred[-val_set:]
    y_test = aresp[-val_set:]
    #df2 = pd.read_csv(_constants.data_location + 'simple_game_data_leagueId={}.csv'.format(2))
    #x_test, y_testL = forest_training_data(df2)
    #x_test = np.array(x_test, dtype=np.float32)
    #y_test = np.array(y_testL, dtype=np.float32).reshape(len(y_testL), 1)
    build_net(x_train, y_train, 10000, x_test, y_test)
    forest = sk.ensemble.RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=4)
    forest.fit(x_train, y_train)
    print(forest.oob_score_, forest.score(x_test, y_test))

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
    #predict_league(league)











