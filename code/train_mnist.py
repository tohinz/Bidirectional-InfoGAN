# Tensorflow Version: 1.5.0

import os
import sys
import datetime
import dateutil.tz
import argparse
from shutil import copyfile

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import *
from visualization import *

np.random.seed(12345)
tf.set_random_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="The size of the minibatch", type=int, default=128)
parser.add_argument("--lr_d", help="Discriminator Learning Rate", type=float, default=1e-4)
parser.add_argument("--lr_g", help="Generator Learning Rate", type=float, default=5e-4)
parser.add_argument("--beta1_g", help="Generator Beta 1 (for Adam Optimizer)", type=float, default=0.5)
parser.add_argument("--beta1_d", help="Discriminator Beta 1 (for Adam Optimizer)", type=float, default=0.5)
parser.add_argument("--num_z", help="Number of noise variables", type=int, default=16)
parser.add_argument("--d_activation", help="Activation function of Discriminator", type=str, default="elu")
parser.add_argument("--g_activation", help="Activation function of Generator", type=str, default="elu")
parser.add_argument("--e_activation", help="Activation function of Encoder", type=str, default="elu")
parser.add_argument("--lr_red", help="Reduction in Learning Rate", type=float, default=5.0)

args = parser.parse_args()

mnist = input_data.read_data_sets('../../MNIST_data', validation_size=5000, one_hot=True)
mnist_validation = mnist.validation.images
mnist_validation = np.reshape(mnist_validation, [5000, 28, 28, 1])

# Hyperparameters
BATCH_SIZE = args.batch_size
MAX_ITER = 30000                    # maximum number of iterations
IMG_WIDTH, IMG_HEIGHT = 28, 28      # image dimensions
IMG_CHANNELS = 1                    # image channels
LR_D = args.lr_d                    # learning rate discriminator
LR_G = args.lr_g                    # learning rate generator and encoder
FINAL_LR_D = LR_D / args.lr_red     # final learning rate discriminator
FINAL_LR_G = LR_G / args.lr_red     # final learning rate generator and encoder
decay_steps_d = MAX_ITER            # number of steps over which the learning rate is decayed (discriminator)
decay_steps_g = MAX_ITER            # number of steps over which the learning rate is decayed (generator and encoder)
BETA1_D = args.beta1_d              # beta1 value for Adam optimizer (discriminator)
BETA1_G = args.beta1_g              # beta1 value for Adam optimizer (generator and encoder)

Z_DIM = args.num_z                  # dimensionality of the z vector (input to G, incompressible noise)
CONT_VARS = 2                       # number of continuous variables
DISC_VARS = [10]                    # number of categorical classes per categorical variable

num_disc_vars = 0
for cla in DISC_VARS:
    num_disc_vars += cla            # total number of discrete variables
C_DIM = num_disc_vars + CONT_VARS   # dimensionality of the c vector (input to G, categorical and continuous variables)


now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

log_dir = "log_dir/mnist/mnist_" + timestamp

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(log_dir+"/samples_cont"):
    os.makedirs(log_dir+"/samples_cont")
if not os.path.exists(log_dir+"/samples_disc"):
    os.makedirs(log_dir+"/samples_disc")

with open(log_dir + "/hyperparameters_"+timestamp+".csv", "wb") as f:
    for arg in args.__dict__:
        f.write(arg + "," + str(args.__dict__[arg]) + "\n")
    f.write("num categorical classes," + str(DISC_VARS) + "\n")
    f.write("num categorical," + str(num_disc_vars) + "\n")
    f.write("num continuous," + str(CONT_VARS) + "\n")

copyfile(sys.argv[0], log_dir + "/" + sys.argv[0])


# activation functions for the Generator, Discriminator, and Encoder
activations = {"elu" : tf.nn.elu, "relu": tf.nn.relu, "lrelu": tf.nn.leaky_relu}
g_activation = activations[args.g_activation]
e_activation = activations[args.e_activation]
d_activation = activations[args.d_activation]


# placeholder variables
X = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], name="X") # real images
z = tf.placeholder(tf.float32, shape=[None, Z_DIM], name="z") # incompressible noise, input to G
c = tf.placeholder(tf.float32, shape=[None, C_DIM], name="c") # disentangled representations, input to G
phase = tf.placeholder(tf.bool, name='phase') # training or inference


def conv2d_bn_act(inputs, filters, kernel_size, kernel_init, activation, strides,
                  noise=False, noise_std=0.5, padding="VALID"):
    """
    Shortcut for a module of convolutional layer, batch normalization and possibly adding of Gaussian noise.
    :param inputs: input data
    :param filters: number of convolutional filters
    :param kernel_size: size of filters
    :param kernel_init: weight initialization
    :param activation: activation function (applied after batch normalization)
    :param strides: strides of the convolutional filters
    :param noise: whether to add gaussian noise to the output
    :param noise_std: stadnard deviation of added noise
    :param padding: padding in the conv layer
    :return: output data after applying the conv layer, batch norm, activation function and possibly Gaussian noise
    """
    _tmp = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            kernel_initializer=kernel_init, activation=None, strides=strides, padding=padding)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase)
    _tmp = activation(_tmp)
    if noise:
        _tmp = gaussian_noise_layer(_tmp, noise_std, phase)

    return _tmp


def deconv2d_bn_act(inputs, filters, kernel_size, kernel_init, activation, strides, padding="SAME"):
    """
        Shortcut for a module of transposed convolutional layer, batch normalization.
        :param inputs: input data
        :param filters: number of convolutional filters
        :param kernel_size: size of filters
        :param kernel_init: weight initialization
        :param activation: activation function (applied after batch normalization)
        :param strides: strides of the convolutional filters
        :param padding: padding in the conv layer
        :return: output data after applying the transposed conv layer, batch norm, and activation function
        """
    _tmp = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            kernel_initializer=kernel_init, activation=None, strides=strides, padding=padding)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase)
    _tmp = activation(_tmp)

    return _tmp


def dense_bn_act(inputs, units, activation, kernel_init, noise=False, noise_std=0.5):
    """
        Shortcut for a module of dense layer, batch normalization and possibly adding of Gaussian noise.
        :param inputs: input data
        :param units: number of units
        :param activation: activation function (applied after batch normalization)
        :param kernel_init: weight initialization
        :return: output data after applying the dense layer, batch norm, activation function and possibly Gaussian noise
        """
    _tmp = tf.layers.dense(inputs=inputs, units=units, activation=None, kernel_initializer=kernel_init)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase)
    _tmp = activation(_tmp)
    if noise:
        _tmp = gaussian_noise_layer(_tmp, noise_std, phase)

    return _tmp


# Discriminator
def discriminate(img_input, noise_input):
    with tf.variable_scope("d_net", reuse=tf.AUTO_REUSE):
        # image discriminator
        d_x_conv_0 = conv2d_bn_act(inputs=img_input, filters=64, kernel_size=3, kernel_init=he_init,
                                   activation=d_activation, strides=2, noise=True, noise_std=0.3)

        d_x_conv_1 = conv2d_bn_act(inputs=d_x_conv_0, filters=128, kernel_size=3, kernel_init=he_init,
                                   activation=d_activation, strides=2, noise=True)

        shp = [int(s) for s in d_x_conv_1.shape[1:]]
        d_x_conv_1 = tf.reshape(d_x_conv_1, [-1, shp[0] * shp[1] * shp[2]])

        d_x_dense = dense_bn_act(inputs=d_x_conv_1, units=512, activation=d_activation, kernel_init=he_init,
                                 noise=True)

        # noise discriminator
        noise_input = tf.reshape(noise_input, (-1, 1, 1, Z_DIM+C_DIM))
        d_z_conv_0 = conv2d_bn_act(inputs=noise_input, filters=64, kernel_size=1, kernel_init=he_init,
                                   activation=d_activation, strides=1, noise=True, noise_std=0.3)

        d_z_conv_1 = conv2d_bn_act(inputs=d_z_conv_0, filters=128, kernel_size=1, kernel_init=he_init,
                                   activation=d_activation, strides=1, noise=True)
        shp = [int(s) for s in d_z_conv_1.shape[1:]]
        d_z_conv_1 = tf.reshape(d_z_conv_1, [-1, shp[0] * shp[1] * shp[2]])

        d_z_dense = dense_bn_act(inputs=d_z_conv_1, units=512, activation=d_activation, kernel_init=he_init,
                                 noise=True)

        # final discriminator
        inp = tf.concat((d_x_dense, d_z_dense), axis=1)
        d_final_dense = dense_bn_act(inputs=inp, units=1024, activation=d_activation, kernel_init=he_init,
                                 noise=True)

        # final prediction whether input is real or generated
        d_final_pred = tf.layers.dense(inputs=d_final_dense, units=1, activation=tf.nn.sigmoid,
                                       kernel_initializer=he_init)
        return d_final_pred


# Encoder + Mutual Information
def encode(image_input):
    with tf.variable_scope("e_net", reuse=tf.AUTO_REUSE):
        e_conv_0 = conv2d_bn_act(inputs=image_input, filters=32, kernel_size=3, kernel_init=he_init,
                                   activation=e_activation, strides=1)

        e_conv_1 = conv2d_bn_act(inputs=e_conv_0, filters=64, kernel_size=3, kernel_init=he_init,
                                 activation=e_activation, strides=2)

        e_conv_2 = conv2d_bn_act(inputs=e_conv_1, filters=128, kernel_size=3, kernel_init=he_init,
                                 activation=e_activation, strides=2)

        shp = [int(s) for s in e_conv_2.shape[1:]]
        e_conv_2 = tf.reshape(e_conv_2, [-1, shp[0] * shp[1] * shp[2]])

        e_dense_0 = dense_bn_act(inputs=e_conv_2, units=1024, activation=e_activation, kernel_init=he_init)

        # prediction for z, i.e. the noise part of the representation
        e_dense_z = tf.layers.dense(inputs=e_dense_0, units=Z_DIM, activation=tf.nn.tanh, kernel_initializer=he_init)

        # prediction for categorical variables of c
        e_dense_c_disc = []
        for idx, classes in enumerate(DISC_VARS):
            e_dense_c_disc.append(tf.layers.dense(inputs=e_dense_0, units=classes, activation=tf.nn.softmax,
                                                  kernel_initializer=he_init, name="e_dense_c_disc_" + str(idx)))
        e_dense_c_disc_concat = tf.concat(e_dense_c_disc, axis=1)

        # prediction for continuous variables of c
        e_dense_c_cont = tf.layers.dense(inputs=e_dense_0, units=CONT_VARS, activation=None,
                                         kernel_initializer=he_init, name="e_dense_c_cont")

        return tf.concat([e_dense_z, e_dense_c_disc_concat, e_dense_c_cont], axis=1)


# Generator
def generate(noise_input):
    with tf.variable_scope("g_net", reuse=tf.AUTO_REUSE):
        g_dense_0 = dense_bn_act(inputs=noise_input, units=3136, activation=g_activation, kernel_init=truncated_normal)
        g_dense_0 = tf.reshape(g_dense_0, [-1, 7, 7, 64])

        g_conv_0 = deconv2d_bn_act(inputs=g_dense_0, filters=128, kernel_size=4, kernel_init=truncated_normal,
                                   activation=g_activation, strides=2)

        g_conv_1 = deconv2d_bn_act(inputs=g_conv_0, filters=64, kernel_size=4, kernel_init=truncated_normal,
                                   activation=g_activation, strides=1)

        g_conv_out = tf.layers.conv2d_transpose(inputs=g_conv_1, filters=1, kernel_size=4, activation=tf.nn.sigmoid,
                                                padding='SAME', strides=2,
                                                kernel_initializer=truncated_normal)
        return g_conv_out


# encoding of an image by the encoder E
Z_hat = encode(X)
# image generated by generator G
X_hat = generate(tf.concat((z, c), axis=1))

# prediction of D for real images with encoding by E
D_enc = discriminate(X, Z_hat)
# prediction of D for generated images
D_gen = discriminate(X_hat, tf.concat((z, c), axis=1))


# minimize crossentropy between z and E(G(z))
# encoding by E on generated images
Z_gen = encode(X_hat)
# get disentangled part of the encoding
c_gen = Z_gen[:, Z_DIM:]
# crossentropy in continuous variables
cont_stddev_c_gen = tf.ones_like(c_gen[:, num_disc_vars:])
eps_c_gen = (c[:, num_disc_vars:] - c_gen[:, num_disc_vars:]) / (cont_stddev_c_gen + 1e-8)
crossent_c_gen_cont = tf.reduce_mean(
    -tf.reduce_sum(-0.5 * np.log(2 * np.pi) - log(cont_stddev_c_gen) - 0.5 * tf.square(eps_c_gen), 1))
# crossentropy in categorical variables
crossent_c_gen_cat = tf.reduce_mean(-tf.reduce_sum(log(c_gen[:, :num_disc_vars]) * c[:, :num_disc_vars], 1))


# Discriminator loss
D_loss = -tf.reduce_mean(log(D_enc) + log(1 - D_gen))
# Generator / Encoder loss
G_loss = -tf.reduce_mean(log(D_gen) + log(1 - D_enc)) + crossent_c_gen_cont + crossent_c_gen_cat

all_vars = tf.trainable_variables()
theta_G_E = [var for var in all_vars if var.name.startswith('g_') or var.name.startswith('e_')]
theta_D = [var for var in all_vars if var.name.startswith('d_')]

# Define the optimizers for D, G, and E
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step, important for updating
    # the batch normalization parameters
    global_step_d = tf.Variable(0, trainable=False)
    lr_dis = tf.train.polynomial_decay(LR_D, global_step_d,
                                       decay_steps_d, FINAL_LR_D,
                                       power=0.5, name="learning_rate_d")
    D_solver = (tf.train.AdamOptimizer(learning_rate=lr_dis, beta1=BETA1_D)
                .minimize(D_loss, var_list=theta_D, global_step=global_step_d))

    global_step_g = tf.Variable(0, trainable=False)
    lr_gen = tf.train.polynomial_decay(LR_G, global_step_g,
                                       decay_steps_g, FINAL_LR_G,
                                       power=0.5, name="learning_rate_g")
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr_gen, beta1=BETA1_G)
                .minimize(G_loss, var_list=theta_G_E, global_step=global_step_g))

# summaries for visualization in Tensorboard
tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
tf.summary.scalar("D data accuracy", tf.reduce_mean(D_enc))
tf.summary.scalar("D fake accuracy", tf.reduce_mean(1 - D_gen))
tf.summary.scalar("c gen cont", crossent_c_gen_cont)
tf.summary.scalar("c gen cat", crossent_c_gen_cat)
summary_op_scalar = tf.summary.merge_all()
summary_ops_cat = summary_cat(X_hat)
summary_ops_cont = summary_cont(X_hat)
summary_op_reconstruction = summary_recon(X, X_hat)

print("Initialize new session")
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=2)
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

print("Start training")
for iteration in range(1, MAX_ITER + 1):
    # sample next MNIST batch
    X_mb, _ = mnist.train.next_batch(BATCH_SIZE)
    X_mb = np.reshape(X_mb, [-1, 28, 28, 1])
    # sample z and c vectors as input to G
    z_mb = sample_z(BATCH_SIZE, Z_DIM)
    c_mb = sample_c(BATCH_SIZE)

    # train D, G, and E
    _, G_loss_curr, _, D_loss_curr = sess.run([G_solver, G_loss, D_solver, D_loss],
                                              feed_dict={X: X_mb, z: z_mb, c: c_mb, 'phase:0': 1})

    # visualize training progress
    if iteration % 5000 == 0:
        print("Iteration: {}; D_loss: {:.4}; G_loss: {:.4}".format(iteration, D_loss_curr, G_loss_curr))

        # visualize scalars
        summary = sess.run(summary_op_scalar, feed_dict={X: X_mb, z: z_mb, c: c_mb, 'phase:0': 1})

        # visualize categorical variables via Tensorboard
        summary_cat = []
        for idx in range(len(DISC_VARS)):
            z_mb = sample_z_fixed(128, Z_DIM)
            c_test = sample_c_cat(128, disc_var=idx)
            z_tmp, _, _summary_cat = sess.run([Z_hat, X_hat, summary_ops_cat[idx]],
                                           feed_dict={X: X_mb, z: z_mb, c: c_test, 'phase:0': 0})
            summary_cat.append(_summary_cat)

        # visualize continuous variables via Tensorboard
        summary_cont = []
        for idx in range(CONT_VARS):
            z_mb = sample_z_fixed(128, Z_DIM)
            c_const = [_ for _ in range(CONT_VARS) if _ != idx]
            c_test = sample_c_cont(c_var=idx, c_const=c_const)
            _, _, _summary_cont = sess.run([Z_hat, X_hat, summary_ops_cont[idx]],
                                       feed_dict={X: X_mb, z: z_mb, c: c_test, 'phase:0': 0})
            summary_cont.append(_summary_cont)

        # visualize reconstruction G(E(x)) via Tensorboard
        z_tmp = np.asarray(z_tmp)
        _, _, summary_recon = sess.run([Z_hat, X_hat, summary_op_reconstruction], feed_dict={X: X_mb, z: z_tmp[:, :Z_DIM],
                                                                              c: z_tmp[:, Z_DIM:], 'phase:0': 0})

        # sample images from test set according to their encodings
        encodings = np.zeros((5000, num_disc_vars + CONT_VARS + Z_DIM))
        for idx_tmp in range(5):
            x_test = mnist_validation[idx_tmp * 1000:idx_tmp * 1000 + 1000]
            encodings_tmp = sess.run(Z_hat, feed_dict={X: x_test, 'phase:0': 0})
            encodings[idx_tmp * 1000:idx_tmp * 1000 + 1000] = encodings_tmp
        sample_cont_test_set(encodings, iteration, mnist_validation, log_dir)
        sample_disc_test_set(encodings, iteration, mnist_validation, log_dir, Z_DIM)

        summary_writer.add_summary(summary, iteration)
        for summ in summary_cat:
            summary_writer.add_summary(summ, iteration)
        for summ in summary_cont:
            summary_writer.add_summary(summ, iteration)
        summary_writer.add_summary(summary_recon, iteration)
        summary_writer.flush()

    # save model
    if iteration % 10000 == 0:
        snapshot_name = "iteration"
        fn = saver.save(sess, "{}/iteration.ckpt".format(log_dir), global_step=iteration)
        print("Model saved in file: {}".format(fn))
