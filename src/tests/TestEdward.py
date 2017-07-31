import tensorflow as tf
import edward as ed
import numpy as np
import Nets


# Gaussian variational 'posterior' with tf.Variable parameters - to be fit with the true posterior
def gauss_var_post(shape):
    return ed.models.Normal(loc=tf.Variable(tf.random_normal(shape)), scale=tf.nn.softplus(tf.Variable(tf.zeros(shape))))


# guassian prior (with requested shape)
def gauss_prior(shape):
    return ed.models.Normal(loc=tf.zeros(shape), scale=tf.ones(shape))



def build_toy_dataset(num):
    x = np.random.randn(num, 2) * 1
    w = np.array([1, 0])
    temp = np.sin(np.dot(x, w))
    out = temp > x.transpose()[1]
    out = np.reshape(out, [num, 1])
    out = out.astype(np.float32)
    return x, out


def train_model(x_train, y_train):
    # x = tf.placeholder(tf.float32, shape=[None, 2])
    # widths of fully-connected layers in NN
    inputs = 2
    outputs = 1
    layer_widths = [8, 8, 8, 8, 8]
    activations = [tf.nn.elu for _ in layer_widths] + [tf.identity]
    layer_widths += [outputs]
    # Input data goes here (via feed_dict or equiv)
    x = tf.placeholder(tf.float32, shape=[len(x_train), inputs])
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
    print('accuracy, log_likelihood, crossentropy', ed.evaluate(['accuracy', 'log_likelihood', 'crossentropy'], data={out: y_train, x: x_train}))
    # Run variational inference, minimizing KL(q, p) using stochastic gradient descent over variational params
    inference = ed.KLqp(var_post, data={out: y_train, x: x_train})
    inference.run(n_samples=16, n_iter=10000)


    # Get output object dependant on variational posteriors rather than priors
    out_post = ed.copy(out, var_post)
    # Re-evaluate metrics
    print('accuracy, log_likelihood, crossentropy', ed.evaluate(['accuracy', 'log_likelihood', 'crossentropy'], data={out_post: y_train, x: x_train}))

ed.set_seed(42)
xr, yr = build_toy_dataset(512)

print(np.shape(yr))
train_model(xr, yr)
