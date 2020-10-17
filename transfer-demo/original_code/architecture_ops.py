import tensorflow as tf
import numpy as np
from utils import wrappy


def _conv(inputs, filters, kernel_size=1, strides=1, pad="VALID", name="conv"):
    with tf.variable_scope(name):
        kernel = tf.get_variable(
            shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            name="weights",
        )
        conv = tf.nn.conv2d(
            inputs, kernel, [1, strides, strides, 1], padding=pad, data_format="NHWC"
        )
        return conv


def _conv_bn_relu(
    inputs, filters, train, kernel_size=1, strides=1, pad="VALID", name="conv_bn_relu"
):
    with tf.variable_scope(name):
        kernel = tf.get_variable(
            shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            name="weights",
        )
        conv = tf.nn.conv2d(
            inputs, kernel, [1, strides, strides, 1], padding=pad, data_format="NHWC"
        )
        norm = tf.nn.relu(
            tf.layers.batch_normalization(
                conv, momentum=0.9, epsilon=1e-5, training=train, name="bn"
            )
        )
        return norm


def _conv_block(inputs, numOut, train, name="conv_block"):
    with tf.variable_scope(name):
        with tf.variable_scope("norm_1"):
            norm_1 = tf.nn.relu(
                tf.layers.batch_normalization(
                    inputs, momentum=0.9, epsilon=1e-5, training=train, name="bn"
                )
            )

            conv_1 = _conv(
                norm_1,
                int(numOut / 2),
                kernel_size=1,
                strides=1,
                pad="VALID",
                name="conv",
            )
        with tf.variable_scope("norm_2"):
            norm_2 = tf.nn.relu(
                tf.layers.batch_normalization(
                    conv_1, momentum=0.9, epsilon=1e-5, training=train, name="bn"
                )
            )

            pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name="pad")
            conv_2 = _conv(
                pad, int(numOut / 2), kernel_size=3, strides=1, pad="VALID", name="conv"
            )
        with tf.variable_scope("norm_3"):
            norm_3 = tf.nn.relu(
                tf.layers.batch_normalization(
                    conv_2, momentum=0.9, epsilon=1e-5, training=train, name="bn"
                )
            )

            conv_3 = _conv(
                norm_3, int(numOut), kernel_size=1, strides=1, pad="VALID", name="conv"
            )
        return conv_3


def _skip_layer(inputs, num_out, name="skip_layer"):
    with tf.variable_scope(name):
        if inputs.get_shape().as_list()[3] == num_out:
            return inputs
        else:
            conv = _conv(inputs, num_out, kernel_size=1, strides=1, name="conv")
            return conv


def _residual(inputs, num_out, train, name="residual_block"):
    with tf.variable_scope(name):
        convb = _conv_block(inputs, num_out, train=train)
        skipl = _skip_layer(inputs, num_out)
        return tf.add_n([convb, skipl], name="res_block")


@wrappy
def nccuc(input_A, input_B, n_filters, padding, training, name):
    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            if i < 1:
                x0 = input_A
                x1 = tf.layers.conv2d(
                    x0,
                    F,
                    (4, 4),
                    strides=(1, 1),
                    activation=None,
                    padding=padding,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                    name="conv_{}".format(i + 1),
                )
                x1 = tf.layers.batch_normalization(
                    x1, training=training, name="bn_{}".format(i + 1)
                )
                x1 = tf.nn.relu(x1, name="relu{}_{}".format(name, i + 1))

            elif i == 1:
                up_conv = tf.layers.conv2d_transpose(
                    x1,
                    filters=F,
                    kernel_size=4,
                    strides=2,
                    padding=padding,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                    name="upsample_{}".format(name),
                )

                up_conv = tf.nn.relu(up_conv, name="relu{}_{}".format(name, i + 1))
                return tf.concat(
                    [up_conv, input_B], axis=-1, name="concat_{}".format(name)
                )

            else:
                return x1
