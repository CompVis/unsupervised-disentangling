from utils import wrappy
import tensorflow as tf
from architecture_ops import _residual, _conv_bn_relu, _conv, nccuc
from ops import softmax


@wrappy
def discriminator_patch(image, train):
    padding = "VALID"
    x0 = image
    x1 = tf.layers.conv2d(
        x0,
        32,
        4,
        strides=1,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_0",
    )  # 46
    x1 = tf.layers.batch_normalization(x1, training=train, name="bn_0")
    x1 = tf.layers.conv2d(
        x1,
        64,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_1",
    )  # 44
    x1 = tf.layers.batch_normalization(x1, training=train, name="bn_1")
    x2 = tf.layers.conv2d(
        x1,
        128,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_2",
    )  # 10
    x2 = tf.layers.batch_normalization(x2, training=train, name="bn_2")
    x3 = tf.layers.conv2d(
        x2,
        256,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_3",
    )  # 4
    x3 = tf.layers.batch_normalization(x3, training=train, name="bn_3")
    x4 = tf.reshape(x3, shape=[-1, 4 * 4 * 256])
    x4 = tf.layers.dense(x4, 1, name="last_fc")
    return tf.nn.sigmoid(x4), x4


@wrappy
def decoder(encoding_list, train, reconstr_dim, n_reconstruction_channels):
    """
    :param encoding_list:
        list of feature maps at each stage to merge.
        For `reconstr_dim = 128?` this is something like
        encoding_list = [
            tf.zeros((1, 128 // (2 ** i), 128 // (2 ** i), 128)) for i in range(6)
        ]
    :param train:
    :param reconstr_dim:
    :param n_reconstruction_channels:
    :return:
    """
    padding = "SAME"

    input = encoding_list[-1]  # 128 channels
    conv1 = nccuc(
        input, encoding_list[-2], [512, 512], padding, train, name=1
    )  # 8, 64 channels
    conv2 = nccuc(
        conv1, encoding_list[-3], [512, 256], padding, train, name=2
    )  # 16, 384 channels
    conv3 = nccuc(conv2, encoding_list[-4], [256, 256], padding, train, name=3)  # 32

    if reconstr_dim == 128:
        conv4 = nccuc(
            conv3, encoding_list[-5], [256, 128], padding, train, name=4
        )  # 64
        conv5 = nccuc(
            conv4, encoding_list[-6], [128, 64], padding, train, name=5
        )  # 128
        conv6 = tf.layers.conv2d(
            conv5,
            n_reconstruction_channels,
            6,
            strides=1,
            padding="SAME",
            activation=tf.nn.sigmoid,
            name="conv_6",
        )
        reconstruction = conv6  # 128

    if reconstr_dim == 256:
        conv4 = nccuc(
            conv3, encoding_list[-5], [256, 128], padding, train, name=4
        )  # 64
        conv5 = nccuc(
            conv4, encoding_list[-6], [128, 128], padding, train, name=5
        )  # 128
        conv6 = nccuc(
            conv5, encoding_list[-7], [128, 64], padding, train, name=6
        )  # 256
        conv7 = tf.layers.conv2d(
            conv6,
            n_reconstruction_channels,
            6,
            strides=1,
            padding="SAME",
            activation=tf.nn.sigmoid,
            name="conv_7",
        )
        reconstruction = conv7  # 256
    return reconstruction


def _hourglass(inputs, n, numOut, train, name="hourglass"):
    """ Hourglass Module
    Args:
        inputs	: Input Tensor
        n		: Number of downsampling step
        numOut	: Number of Output Features (channels)
        name	: Name of the block
    """
    with tf.variable_scope(name):
        # Upper Branch
        up_1 = _residual(inputs, numOut, train=train, name="up_1")
        # Lower Branch
        low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding="VALID")
        low_1 = _residual(low_, numOut, train=train, name="low_1")

        if n > 0:
            low_2 = _hourglass(low_1, n - 1, numOut, train=train, name="low_2")
        else:
            low_2 = _residual(low_1, numOut, train=train, name="low_2")

        low_3 = _residual(low_2, numOut, train=train, name="low_3")
        up_2 = tf.image.resize_nearest_neighbor(
            low_3, tf.shape(low_3)[1:3] * 2, name="upsampling"
        )
        return tf.add_n([up_2, up_1], name="out_hg")


@wrappy
def seperate_hourglass(inputs, train, n_landmark, n_features, nFeat_1, nFeat_2):
    _ , h, w, c = inputs.get_shape().as_list()
    nLow = 4  # hourglass preprocessing reduces by factor two hourglass by factor 16 (2^4)  e.g. 128 -> 4
    n_Low_feat = 1
    dropout_rate = 0.2

    # Storage Table
    hg = [None] * 2
    ll = [None] * 2
    ll_ = [None] * 2
    drop = [None] * 2
    out = [None] * 2
    out_ = [None] * 2
    sum_ = [None] * 2

    with tf.variable_scope("model"):
        with tf.variable_scope("preprocessing"):
            if h == 256:
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name="pad_1")
                conv1 = _conv_bn_relu(
                    pad1,
                    filters=64,
                    train=train,
                    kernel_size=6,
                    strides=2,
                    name="conv_256_to_128",
                )
                r1 = _residual(conv1, num_out=128, train=train, name="r1")
                pool1 = tf.contrib.layers.max_pool2d(
                    r1, [2, 2], [2, 2], padding="VALID"
                )
                r2 = _residual(pool1, num_out=int(nFeat_1 / 2), train=train, name="r2")
                r3 = _residual(r2, num_out=nFeat_1, train=train, name="r3")

            elif h == 128:
                pad1 = tf.pad(
                    inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name="pad_1"
                )  # shape [1, 132, 132, 10]
                conv1 = _conv_bn_relu(
                    pad1,
                    filters=64,
                    train=train,
                    kernel_size=6,
                    strides=2,
                    name="conv_64_to_32",
                )  # shape [1, 64, 64, 64]
                r3 = _residual(conv1, num_out=nFeat_1, train=train, name="r3")
                # shape [1, 64, 64, nFeat_1]
            elif h == 64:
                pad1 = tf.pad(inputs, [[0, 0], [3, 2], [3, 2], [0, 0]], name="pad_1")
                conv1 = _conv_bn_relu(
                    pad1,
                    filters=64,
                    train=train,
                    kernel_size=6,
                    strides=1,
                    name="conv_64_to_32",
                )
                r3 = _residual(conv1, num_out=nFeat_1, train=train, name="r3")

            else:
                raise ValueError

        with tf.variable_scope("stage_0"):
            hg[0] = _hourglass(
                r3, nLow, nFeat_1, train=train, name="hourglass"
            )  # [1, 64, 64, nFeat_1]
            drop[0] = tf.layers.dropout(
                hg[0], rate=dropout_rate, training=train, name="dropout"
            )  # [1, 64, 64, nFeat_1]
            ll[0] = _conv_bn_relu(
                drop[0],
                nFeat_1,
                train=train,
                kernel_size=1,
                strides=1,
                pad="VALID",
                name="conv",
            )  # [1, 64, 64, nFeat_1]
            ll_[0] = _conv(ll[0], nFeat_1, 1, 1, "VALID", "ll")  # [1, 64, 64, nFeat_1]
            out[0] = _conv(
                ll[0], n_landmark, 1, 1, "VALID", "out"
            )  # [1, 64, 64, n_landmark]
            out_[0] = _conv(
                softmax(out[0]), nFeat_1, 1, 1, "VALID", "out_"
            )  # [1, 64, 64, nFeat_1]
            sum_[0] = tf.add_n([out_[0], r3], name="merge")  # [1, 64, 64, nFeat_1]

        with tf.variable_scope("stage_1"):
            hg[1] = _hourglass(
                sum_[0], n_Low_feat, nFeat_2, train=train, name="hourglass"
            )  # [1, 64, 64, nFeat_2]
            drop[1] = tf.layers.dropout(
                hg[1], rate=dropout_rate, training=train, name="dropout"
            )  # [1, 64, 64, nFeat_2]
            ll[1] = _conv_bn_relu(
                drop[1],
                nFeat_2,
                train=train,
                kernel_size=1,
                strides=1,
                pad="VALID",
                name="conv",
            )  # [1, 64, 64, nFeat_2]

            out[1] = _conv(ll[1], n_features, 1, 1, "VALID", "out")
            # [1, 64, 64, n_features]

        features = out[1]  # [1, 64, 64, nFeat_2]
        return softmax(out[0]), features


decoder_map = {"standard": decoder}
encoder_map = {"seperate": seperate_hourglass}
