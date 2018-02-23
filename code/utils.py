import numpy as np
import tensorflow as tf
from tensorflow.python.layers import utils

def log(x):
    return tf.log(x + 1e-8)


def get_max_idx(max, a):
    return [i for i, j in enumerate(a) if j == max][0]


def get_min_idx(min, a):
    return [i for i, j in enumerate(a) if j == min][0]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# weight initialization functions
def he_init(size, dtype=tf.float32, partition_info=None):
    in_dim = size[0]
    he_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=he_stddev)

truncated_normal = tf.truncated_normal_initializer(stddev=0.02)


def gaussian_noise_layer(input_layer, std, training):
    def add_noise():
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    return utils.smart_cond(training, add_noise, lambda: input_layer)


def sample_z_fixed(m, n):
    z_out = np.zeros((m, n))
    for idx in range(10):
        z_out[idx * 10:idx * 10 + 10, :] = np.random.uniform(-1., 1., size=[1, n])
    return z_out


def sample_c(m, num_cont_vars=2, disc_classes=[10]):
    """
    Sample a random c vector, i.e. categorical and continuous variables.
    If test is True, samples a value for each discrete variable and combines each with the chosen continuous
    variable c; all other continuous c variables are sampled once and then kept fixed
    """
    c = np.random.multinomial(1, disc_classes[0] * [1.0 / disc_classes[0]], size=m)
    for cla in disc_classes[1:]:
        c = np.concatenate((c, np.random.multinomial(1, cla * [1.0 / cla], size=m)), axis=1)
    for n in range(num_cont_vars):
        cont = np.random.uniform(-1, 1, size=(m, 1))
        c = np.concatenate((c, cont), axis=1)
    return c


def sample_c_cat(m, disc_var=0, num_cont_vars=2, num_disc_vars=10, disc_classes=[10]):
    """
    Samples categorical values for visualization purposes
    """
    cont = []
    cont_matr = []
    for idx in range(num_cont_vars):
        cont.append(np.random.uniform(-1, 1, size=[1, 10]))
        cont_matr.append(np.zeros(shape=(m)))

    for idx in range(num_cont_vars):
        for idx2 in range(10):
            cont_matr[idx][10 * idx2: 10 * idx2 + 10] = np.broadcast_to(cont[idx][0, idx2], (10))

    cs_cont = np.zeros(shape=(128, num_cont_vars))

    for idx in range(num_cont_vars):
        cs_cont[:, idx] = cont_matr[idx]

    c = np.eye(10, 10)
    for idx in range(1, 10):
        c_tmp = np.eye(10, 10)
        c = np.concatenate((c, c_tmp), axis=0)

    counter = 0
    cs_disc = np.zeros(shape=(100, num_disc_vars))
    for idx, cla in enumerate(disc_classes):
        if idx == disc_var:
            tmp = np.eye(cla, cla)
            tmp_ = np.eye(cla, cla)
            for idx2 in range(100 / cla - 1):
                tmp = np.concatenate((tmp, tmp_), axis=0)
            cs_disc[:, counter:counter + cla] = tmp
            counter += cla
        else:
            rand = np.random.randint(0, cla)
            tmp = np.zeros(shape=(100, cla))
            tmp[:, rand] = 1
            cs_disc[:, counter:counter + cla] = tmp
            counter += cla

    zeros = np.zeros((28, num_disc_vars))
    cs_disc = np.concatenate((cs_disc, zeros), axis=0)

    c = np.concatenate((cs_disc, cs_cont), axis=1)

    return c


def sample_c_cont(num_cont_vars=2, num_disc_vars=10, c_dim=12, c_var=0, c_const=[1]):
    """
    Samples continuous values for visualization purposes
    """
    z = []
    for idx in range(len(c_const)):
        z.append(np.random.uniform(-1, 1))
    cont = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0]

    c_cont = np.zeros((10, num_cont_vars))
    i = 0
    for idx in range(num_cont_vars):
        if idx == c_var:
            c_cont[:, c_var] = cont
        else:
            c_cont[:, idx] = z[i]
            i += 1

    c_out = np.zeros((128, c_dim))
    for idx in range(10):
        c_out[10 * idx:10 * idx + 10, idx] = 1
        c_out[10 * idx:10 * idx + 10, num_disc_vars:] = c_cont

    return c_out
