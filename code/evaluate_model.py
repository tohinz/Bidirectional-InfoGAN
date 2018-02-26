# Tensorflow Version: 1.5.0

import os
import sys
import argparse
import csv
from random import Random

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import *
from visualization import *


np.random.seed(12345)
tf.set_random_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", help="Direcotry where the model weights are stored.", type=str)
parser.add_argument("--iteration", help="Model iteration to use.", type=int, default=30000)
parser.add_argument("--generate", help="Generate new images", action='store_true')
parser.add_argument("--evaluate", help="Evaluate accuracy on the test set", action='store_true')
parser.add_argument("--sample", help="Sample from the test set", action='store_true')
parser.add_argument("--reconstruct", help="Sample from the test set and reconstruct with the Encoder and Generator",
                    action='store_true')
args = parser.parse_args()

model_dir = args.model_dir+"/iteration.ckpt-"+str(args.iteration)

assert os.path.exists(model_dir+".meta"), "There exists no weight file \"iteration.ckpt-"+str(args.iteration) + "\" in {}". \
        format(args.model_dir)


def read_encodings(path):
    with open(path+"/encodings.txt") as f:
        encodings = map(int, f.read().strip().split(','))

    return encodings


def read_hyperparameters(path):
    reader = csv.reader(open(path+"/hyperparameters"+args.model_dir[-20:]+".csv", "rb"))
    dict = {}
    for row in reader:
        k, v = row
        dict[k] = v

    return dict

hp_dict = read_hyperparameters(args.model_dir)

activations = {"elu" : tf.nn.elu, "relu": tf.nn.relu, "lrelu": tf.nn.leaky_relu}

g_activation = activations[hp_dict["g_activation"]]
e_activation = activations[hp_dict["e_activation"]]
d_activation = activations[hp_dict["d_activation"]]


Z_DIM = int(hp_dict["num_z"])
DISC_VARS = [10]
num_disc_vars = 0
for cla in DISC_VARS:
    num_disc_vars += cla
CONT_VARS = 2
C_DIM = num_disc_vars + CONT_VARS

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

truncated_normal = tf.truncated_normal_initializer(stddev=0.02)
random_normal = tf.random_normal_initializer(mean=0.0, stddev=0.01)


X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="X")
Y = tf.placeholder(tf.float32, shape=[None, 10], name="Y")
z = tf.placeholder(tf.float32, shape=[None, Z_DIM], name="z")
c = tf.placeholder(tf.float32, shape=[None, C_DIM], name="c")
phase = tf.placeholder(tf.bool, name='phase')


def sample_img_idxs(num_samples):
    # sample num_samples images of each class from the MNIST test set
    py_rng = Random()
    idxs = np.arange(len(mnist.test.images))
    classes_idxs = [idxs[np.argmax(mnist.test.labels, axis=1) == y] for y in range(10)]
    sampled_idxs = [py_rng.sample(class_idxs, num_samples) for class_idxs in classes_idxs]
    sampled_idxs = np.asarray(sampled_idxs).flatten()
    return sampled_idxs


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


### don't train a classifier, just use c_disc directly as label
def test_set_accuracy(keys, encodings, z_test=args.model_dir + "/z_representations_test.npy"):
    z_representations_test = np.load(z_test)
    mappings = dict(zip(encodings, keys))
    predictions = []
    for idx in range(z_representations_test.shape[0]):
        predictions.append(mappings[np.argmax(z_representations_test[idx, Z_DIM:Z_DIM + num_disc_vars])])
    predictions = np.asarray(predictions)

    Y_mb = mnist.test.labels
    mnist_labels = np.argmax(Y_mb, axis=1)

    correct_predictions = np.equal(predictions, mnist_labels)
    acc = np.mean(correct_predictions)

    print("Accuracy: {}".format(acc))


### save z representations, i.e. encodings of images
def sample_z_reps(train=False):
    repr = encode(X)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_dir)

    if train and not os.path.exists(args.model_dir + "/z_representations_train.npy"):
        z_representations = []
        # save train representations
        for idx in range(55):
            X_mb, _ = mnist.train.next_batch(1000, shuffle=False)
            X_mb = np.reshape(X_mb, [-1, 28, 28, 1])

            z_rep = sess.run(repr, feed_dict={X: X_mb, phase: 0})
            z_rep = np.asarray(z_rep)
            z_representations.append(z_rep)

        z_representations = np.asarray(z_representations)
        with open(args.model_dir + "/z_representations_train.npy", "wb") as f:
            np.save(f, z_representations)

    # save test representations
    if not os.path.exists(args.model_dir + "/z_representations_test.npy"):
        z_representations = []
        for idx in range(10):
            # X_mb, _ = mnist.test.next_batch(1000, shuffle=False)
            X_mb = mnist.test.images[idx * 1000:idx * 1000 + 1000]
            X_mb = np.reshape(X_mb, [-1, 28, 28, 1])

            z_rep = sess.run(repr, feed_dict={X: X_mb, phase: 0})
            z_rep = np.asarray(z_rep)
            z_representations.append(z_rep)

        z_representations = np.asarray(z_representations)
        z_representations = np.reshape(z_representations, (10000, Z_DIM + C_DIM))
        with open(args.model_dir + "/z_representations_test.npy", "wb") as f:
            np.save(f, z_representations)


### sample images with low / high values of continuous variables
def sample_continuous(keys, encodings, z_test=args.model_dir+"/z_representations_test.npy"):
    if not os.path.exists(args.model_dir + "/samples"):
        os.makedirs(args.model_dir + "/samples")
    z_representations_test = np.load(z_test)

    mappings = dict(zip(encodings, keys))

    num_imgs = 30
    max_c1 = np.zeros((10,num_imgs))
    max_c1_idx = np.zeros((10,num_imgs))
    min_c1 = np.ones((10,num_imgs))
    min_c1_idx = np.zeros((10,num_imgs))
    max_c2 = np.zeros((10,num_imgs))
    max_c2_idx = np.zeros((10,num_imgs))
    min_c2 = np.ones((10,num_imgs))
    min_c2_idx = np.zeros((10,num_imgs))
    for idx, rep in enumerate(z_representations_test):
        label = mappings[np.argmax(rep[Z_DIM:Z_DIM+num_disc_vars])]
        act = max(rep[Z_DIM:Z_DIM+num_disc_vars])
        c1 = rep[-2]
        c2 = rep[-1]
        if act > 0.996:
            if c1 > min(max_c1[label]):
                _ = get_min_idx(min(max_c1[label]), max_c1[label])
                max_c1[label, _] = c1
                max_c1_idx[label, _] = idx
            if c1 < max(min_c1[label]):
                _ = get_max_idx(max(min_c1[label]), min_c1[label])
                min_c1[label, _] = c1
                min_c1_idx[label, _] = idx
            if c2 > min(max_c2[label]):
                _ = get_min_idx(min(max_c2[label]), max_c2[label])
                max_c2[label, _] = c2
                max_c2_idx[label, _] = idx
            if c2 < max(min_c2[label]):
                _ = get_max_idx(max(min_c2[label]), min_c2[label])
                min_c2[label, _] = c2
                min_c2_idx[label, _] = idx


    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    imgs = mnist.test.images
    def create_image(min_idx, max_idx, name):
        fig = plt.figure(figsize=(num_imgs*0.28, 5.6))
        gs1 = gridspec.GridSpec(20, num_imgs, wspace=0.0, hspace=0.0)

        for idx in range(20):
            for idx2 in range(num_imgs):
                if idx % 2 == 0:
                    id = int(min_idx[idx/2, idx2])
                else:
                    id = int(max_idx[idx/2, idx2])
                img = imgs[id]
                ax = plt.subplot(gs1[idx*30+idx2])
                ax.imshow(np.reshape(img, [28, 28]), cmap='Greys_r')
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(args.model_dir + "/samples/sample_cont_"+str(name)+".png")

    create_image(min_c1_idx, max_c1_idx, "c1")
    create_image(min_c2_idx, max_c2_idx, "c2")


### sample images from discrete variables
def sample_discrete(keys, encodings, z_test=args.model_dir+"/z_representations_test.npy"):
    if not os.path.exists(args.model_dir + "/samples"):
        os.makedirs(args.model_dir + "/samples")
    z_representations_test = np.load(z_test)

    mappings = dict(zip(encodings, keys))

    num_imgs = 60
    disc_idx = np.zeros((10, num_imgs))

    for idx in range(10):
        for idx2 in range(num_imgs):
            _ = np.random.randint(0, 10000)
            rep = z_representations_test[_, Z_DIM:Z_DIM+num_disc_vars]
            label = mappings[np.argmax(rep)]
            while (not label == idx) or (_ in disc_idx[label]):
                _ = np.random.randint(0, 10000)
                rep = z_representations_test[_, Z_DIM:Z_DIM + num_disc_vars]
                label = mappings[np.argmax(rep)]
            disc_idx[label, idx2] = _


    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    imgs = mnist.test.images

    def create_image():
        fig = plt.figure(figsize=(0.28*30, 0.28*20))
        gs1 = gridspec.GridSpec(20, 30, wspace=0.0, hspace=0.0)
        for idx1 in range(10):
            for idx2 in range(60):
                id = int(disc_idx[idx1, idx2])
                img = imgs[id]
                ax = plt.subplot(gs1[idx1*60+idx2])
                ax.imshow(np.reshape(img, [28, 28]), cmap='Greys_r')
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(args.model_dir + "/samples/samples_categorical.png")

    create_image()


### encode image with E, then decode with G and compare
def reconstruct():
    if not os.path.exists(args.model_dir + "/samples"):
        os.makedirs(args.model_dir + "/samples")

    encoding = encode(X)
    reconstruction = generate(encoding)

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_dir)

    # sample 20 images of each class from the MNIST test set
    sampled_idxs = sample_img_idxs(20)

    imgs = mnist.test.images[sampled_idxs]
    imgs = np.reshape(imgs, (-1, 28, 28, 1))
    imgs_recon = sess.run(reconstruction, feed_dict={X: imgs, phase: 0})


    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    def create_image():
        fig = plt.figure(figsize=(0.28 * 20, 0.28 * 20))
        gs1 = gridspec.GridSpec(20, 20, wspace=0.0, hspace=0.0)
        for idx1 in range(0, 20):
            for idx2 in range(0, 20, 2):
                ax = plt.subplot(gs1[idx1 * 20 + idx2])
                ax.imshow(np.reshape(imgs[idx1 * 10 + idx2/2], [28, 28]), cmap='Greys_r')
                ax.set_xticks([])
                ax.set_yticks([])
                ax = plt.subplot(gs1[idx1 * 20 + idx2+1])
                ax.imshow(np.reshape(imgs_recon[idx1 * 10 + idx2/2], [28, 28]), cmap='Greys_r')
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(args.model_dir + "/samples/samples_reconstruction.png")

    create_image()


def generate_new_samples():
    if not os.path.exists(args.model_dir + "/samples"):
        os.makedirs(args.model_dir + "/samples")

    z_tmp = sample_z_fixed(128, Z_DIM)
    c_tmp = sample_c_cat(128)
    _gen_imgs = generate(tf.concat((z, c), axis=1))

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model_dir)

    gen_imgs= sess.run(_gen_imgs, feed_dict={z: z_tmp[:100], c: c_tmp[:100], phase: 0})

    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    def create_image_categorical():
        fig = plt.figure(figsize=(0.28*10, 0.28*10))
        gs1 = gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)
        for idx1 in range(10):
            for idx2 in range(10):
                img = gen_imgs[idx1*10+idx2]
                ax = plt.subplot(gs1[idx1*10+idx2])
                ax.imshow(np.reshape(img, [28, 28]), cmap='Greys_r')
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(args.model_dir + "/samples/generated_imgs_categorical.png")

    create_image_categorical()


if args.generate:
    generate_new_samples()
elif args.evaluate:
    assert os.path.exists(args.model_dir + '/encodings.txt'), "There exists no file \"encodings.txt\" in {}". \
        format(args.model_dir)
    sample_z_reps()
    keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    encodings = read_encodings(args.model_dir)
    test_set_accuracy(keys, encodings)
elif args.sample:
    sample_z_reps()
    keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    encodings = read_encodings(args.model_dir)
    sample_discrete(keys, encodings)
    sample_continuous(keys, encodings)
elif args.reconstruct:
    reconstruct()
else:
    print("No valid option chosen. Choose either \"--generate\", \"--evaluate\", \"--sample\" or \"--reconstruct\".")
    print("Use \"--help\" for an overview of the command line arguments.")


