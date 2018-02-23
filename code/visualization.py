import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils import *

def summary_cat(gen_imgs, disc_classes=[10]):
    # image logging, discrete variable
    summary_ops_disc = []
    for idx in range(len(disc_classes)):
        stacked_img = []
        for row in xrange(10):
            row_img = []
            for col in xrange(10):
                row_img.append(gen_imgs[(row * 10) + col, :, :, :])
            stacked_img.append(tf.concat(row_img, 1))
        imgs = tf.concat(stacked_img, 0)
        imgs = tf.expand_dims(imgs, 0)
        summary_ops_disc.append(tf.summary.image("image_categorical_" + str(idx), imgs))
    return summary_ops_disc


def summary_cont(gen_imgs, num_cont_vars=2):
    # image logging, continuous variable
    summary_ops_cont = []
    for idx in range(num_cont_vars):
        # gen_imgs = tf.reshape(gen_imgs, [-1, 28, 28, 1])
        stacked_img = []
        for row in xrange(5):
            row_img = []
            for col in xrange(10):
                row_img.append(gen_imgs[(row * 10) + col, :, :, :])
            stacked_img.append(tf.concat(row_img, 1))
        imgs = tf.concat(stacked_img, 0)
        imgs = tf.expand_dims(imgs, 0)
        summary_ops_cont.append(tf.summary.image("image_continuous_" + str(idx), imgs))
    return summary_ops_cont


def summary_recon(real_imgs, recons_imgs):
    # image logging, E(G(z))
    stacked_img = []
    for row in xrange(5):
        row_img = []
        for col in xrange(10):
            row_img.append(real_imgs[(row * 10) + col, :, :, :])
            row_img.append(recons_imgs[(row * 10) + col, :, :, :])
        stacked_img.append(tf.concat(row_img, 1))
    imgs = tf.concat(stacked_img, 0)
    imgs = tf.expand_dims(imgs, 0)
    return tf.summary.image("image_reconstruction", imgs)


def sample_cont_test_set(encodings, it, mnist_test, log_dir, z_dim=16, num_disc_vars=10):
    max_c1 = np.zeros(10)
    max_c1_idx = [0]*10
    min_c1 = np.ones(10)
    min_c1_idx = [0]*10
    max_c2 = np.zeros(10)
    max_c2_idx = [0]*10
    min_c2 = np.ones(10)
    min_c2_idx = [1]*10
    for idx, rep in enumerate(encodings):
        label = np.argmax(rep[z_dim:z_dim+num_disc_vars])
        c1 = rep[-2]
        c2 = rep[-1]
        # if c1 > max_c1[label]:
        if c1 > max_c1[label]:# and c1 < 2:
            max_c1[label] = c1
            max_c1_idx[label] = idx
        # if c1 < min_c1[label]:
        if c1 < min_c1[label]:# and c1 > -2:
            min_c1[label] = c1
            min_c1_idx[label] = idx
        # if c2 > max_c2[label]:
        if c2 > max_c2[label]:# and c2 < 2:
            max_c2[label] = c2
            max_c2_idx[label] = idx
        # if c2 < min_c2[label]:
        if c2 < min_c2[label]:# and c2 > -2:
            min_c2[label] = c2
            min_c2_idx[label] = idx

    f, axarr = plt.subplots(2, 10)
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

    for idx, c1 in enumerate(max_c1_idx):
        img = mnist_test[c1]
        axarr[0, idx].imshow(np.reshape(img, [28, 28]))
        axarr[0, idx].set_xticks([])
        axarr[0, idx].set_yticks([])
        # scipy.misc.imsave('test_imgs/c1_max_' + str(c1) + 'img_' + str(idx) + '.jpg', np.reshape(img, [28, 28]))
    for idx, c1 in enumerate(min_c1_idx):
        img = mnist_test[c1]
        axarr[1, idx].imshow(np.reshape(img, [28, 28]))
        axarr[1, idx].set_xticks([])
        axarr[1, idx].set_yticks([])
        # scipy.misc.imsave('test_imgs/c1_min_' + str(c1) + 'img_' + str(idx) + '.jpg', np.reshape(img, [28, 28]))
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(log_dir + "/samples_cont/" + str(it) + "_c1.png")
    plt.close()

    f, axarr = plt.subplots(2, 10)
    for idx, c2 in enumerate(max_c2_idx):
        img = mnist_test[c2]
        axarr[0, idx].imshow(np.reshape(img, [28, 28]))
        axarr[0, idx].set_xticks([])
        axarr[0, idx].set_yticks([])
        # scipy.misc.imsave('test_imgs/c2_max_' + str(c2) + 'img_' + str(idx) + '.jpg', np.reshape(img, [28, 28]))
    for idx, c2 in enumerate(min_c2_idx):
        img = mnist_test[c2]
        axarr[1, idx].imshow(np.reshape(img, [28, 28]))
        axarr[1, idx].set_xticks([])
        axarr[1, idx].set_yticks([])
        # scipy.misc.imsave('test_imgs/c2_min' + str(c2) + 'img_' + str(idx) + '.jpg', np.reshape(img, [28, 28]))
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(log_dir + "/samples_cont/" + str(it) + "_c2.png")
    plt.close()


def sample_disc_test_set(encodings, it, mnist_test, log_dir, z_dim):
    def _update(d_rep, d_act, d_idx, idx):
        d_class = get_max_idx(max(d_rep), d_rep)
        _d_act = max(d_rep)
        if _d_act > min(d_act[d_class]):
            argmin = np.argmin(d_act[d_class])
            d_act[d_class, argmin] = _d_act
            d_idx[d_class, argmin] = idx
        return d_act, d_idx

    disc1_act, disc1_idx = np.zeros((10, 10)), np.zeros((10, 10))

    for idx, rep in enumerate(encodings):
        rep = rep[z_dim:]
        disc1_act, disc1_idx = _update(rep[:10], disc1_act, disc1_idx, idx)

    def create_image(d_idx, name):
        f, axarr = plt.subplots(10, 10)
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
        for idx1 in range(10):
            for idx2 in range(10):
                id = int(d_idx[idx1, idx2])
                img = mnist_test[id]
                axarr[idx1, idx2].imshow(np.reshape(img, [28, 28]))
                axarr[idx1, idx2].set_xticks([])
                axarr[idx1, idx2].set_yticks([])
        plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(log_dir + "/samples_disc/" + str(it) + "_d" + name + ".png")
        plt.close()

    create_image(disc1_idx, "1")
