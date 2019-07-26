import numpy as np
import tensorflow as tf
from dotmap import DotMap

def tf_rotation_mat(rotation):
    """
    :param rotation: tf tensor of shape [1]
    :return: rotation matrix as tf tensor with shape [2, 2]
    """
    a = tf.expand_dims(tf.cos(rotation), axis=0)
    b = tf.expand_dims(tf.sin(rotation), axis=0)
    row_1 = tf.concat([a, -b], axis=1)
    row_2 = tf.concat([b, a], axis=1)
    mat = tf.concat([row_1, row_2], axis=0)
    return mat

def tps_parameters(batch_size, scal, tps_scal, rot_scal, off_scal, scal_var, rescal=1):
    coord = tf.constant([[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5],
                         [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2], [-0.2, - 0.2]]]
                        , dtype=tf.float32)

    coord = tf.tile(coord, [batch_size, 1, 1])
    shape = coord.get_shape()
    coord = coord + tf.random_uniform(shape=shape, minval=-0.2, maxval=0.2)
    vector = tf.random_uniform(shape=shape, minval=-tps_scal, maxval=tps_scal, dtype=tf.float32)

    offset = tf.random_uniform(shape=[batch_size, 1, 2], minval=-off_scal, maxval=off_scal, dtype=tf.float32)
    offset_2 = tf.random_uniform(shape=[batch_size, 1, 2], minval=-off_scal, maxval=off_scal, dtype=tf.float32)
    t_scal = tf.random_uniform(shape=[batch_size, 2], minval=scal * (1. - scal_var), maxval=scal * (1. + scal_var),
                               dtype=tf.float32)
    t_scal = t_scal * rescal

    rot_param = tf.random_uniform(shape=[batch_size, 1], minval=-rot_scal, maxval=rot_scal, dtype=tf.float32)
    rot_mat = tf.map_fn(tf_rotation_mat, rot_param)

    parameter_dict = {'coord': coord, 'vector': vector, 'offset': offset, 'offset_2': offset_2,
                      't_scal': t_scal, 'rot_mat': rot_mat}
    parameter_dict = DotMap(parameter_dict)
    return parameter_dict


def static_param_2d(param):
    bn, d_1 = param.get_shape().as_list()
    param = param[::2]
    param = tf.tile(param, [1, 2])
    param = tf.reshape(param, [bn, d_1])

    return param


def static_param_3d(param):
    bn, d_1, d_2 = param.get_shape().as_list()
    param = param[::2]
    param = tf.tile(param, [1, 2, 1])
    param = tf.reshape(param, [bn, d_1, d_2])
    return param


def make_input_tps_param(tps_param, move_point=None, scal_point=None):
    coord = tps_param.coord
    vector = tps_param.vector
    offset = tps_param.offset
    offset_2 = tps_param.offset_2
    rot_mat = tps_param.rot_mat
    t_scal = tps_param.t_scal

    scaled_coord = tf.einsum('bk,bck->bck', t_scal, coord + vector - offset) + offset
    t_vector = tf.einsum('blk,bck->bcl', rot_mat, scaled_coord - offset_2) + offset_2 - coord

    if move_point is not None and scal_point is not None:
        coord = tf.einsum('bk,bck->bck', scal_point, coord + move_point)
        t_vector = tf.einsum('bk,bck->bck', scal_point, t_vector)

    else:
        assert(move_point is None and scal_point is None)

    return coord, t_vector


def adapt_tps_for_crop(tps_param, move_point, scal_point):
    """
    :param center_point: b, 1, 2
    :param tps_param:
    :return:
    """
    move_point = - move_point
    scal_point = 1. / scal_point
    crop_coord, t_vector_coord = make_input_tps_param(tps_param, move_point, scal_point)
    return crop_coord, t_vector_coord

#code adapted from https://github.com/iwyoo/tf_ThinPlateSpline

def ThinPlateSpline(U, coord, vector, out_size, n_c, move=None, scal=None):

    coord = coord[:, :, ::-1]
    vector = vector[:, :, ::-1]

    num_batch = tf.shape(U)[0]
    height = tf.shape(U)[1]
    width = tf.shape(U)[2]
    channels = n_c
    out_height = out_size
    out_width = out_size
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    num_point = tf.shape(coord)[1]

    def _repeat(x, n_repeats):
        rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(im, y, x):
        # constants
        y = tf.cast(y, 'float32')
        x = tf.cast(x, 'float32')

        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(height - 1, 'int32')
        max_x = tf.cast(width - 1, 'int32')

        # scale indices from aprox [-1, 1] to [0, width/height]

        y = (y + 1) * height_f / 2.0
        x = (x + 1) * width_f / 2.0

        y = tf.reshape(y, [-1])
        x = tf.reshape(x, [-1])

        # do sampling
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1

        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)

        base = _repeat(tf.range(num_batch) * width * height, out_height * out_width)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, [-1, channels])
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output

    def _meshgrid(height, width, coord):

        x_t = tf.tile(tf.reshape(tf.linspace(- 1., 1., width), [1, width]), [height, 1])
        y_t = tf.tile(tf.reshape(tf.linspace(- 1., 1., height), [height, 1]), [1, width])

        x_t_flat = tf.reshape(x_t, (1, 1, -1))
        y_t_flat = tf.reshape(y_t, (1, 1, -1))

        px = tf.expand_dims(coord[:, :, 0], 2)  # [bn, pn, 1]
        py = tf.expand_dims(coord[:, :, 1], 2)  # [bn, pn, 1]
        d2 = tf.square(x_t_flat - px) + tf.square(y_t_flat - py)
        r = d2 * tf.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = tf.tile(x_t_flat, [num_batch, 1, 1])  # [bn, 1, h*w]
        y_t_flat_g = tf.tile(y_t_flat, [num_batch, 1, 1])  # [bn, 1, h*w]
        ones = tf.ones_like(x_t_flat_g)  # [bn, 1, h*w]

        grid = tf.concat([ones, x_t_flat_g, y_t_flat_g, r], 1)  # [bn, 3+pn, h*w]
        return grid

    def _transform(T, coord, move, scal):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        grid = _meshgrid(out_height, out_width, coord)  # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = tf.matmul(T, grid)  #
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])

        if move is not None and scal is not None:
            off_y = tf.expand_dims(move[:, :, 0], axis=-1)
            off_x = tf.expand_dims(move[:, :, 1], axis=-1)
            scal_y = tf.expand_dims(tf.expand_dims(scal[:, 0], axis=-1), axis=-1)
            scal_x = tf.expand_dims(tf.expand_dims(scal[:, 1], axis=-1), axis=-1)
            y = (y_s * scal_y + off_y)
            x = (x_s * scal_x + off_x)

        else:
            assert (move is None and scal is None)
            y = y_s
            x = x_s

        return y, x

    def _solve_system(coord, vector):
        ones = tf.ones([num_batch, num_point, 1], dtype="float32")
        p = tf.concat([ones, coord], 2)  # [bn, pn, 3]

        p_1 = tf.reshape(p, [num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = tf.reshape(p, [num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = tf.reduce_sum(tf.square(p_1 - p_2), 3)  # [bn, pn, pn]
        r = d2 * tf.log(d2 + 1e-6)  # Kernel [bn, pn, pn]

        zeros = tf.zeros([num_batch, 3, 3], dtype="float32")
        W_0 = tf.concat([p, r], 2)  # [bn, pn, 3+pn]
        W_1 = tf.concat([zeros, tf.transpose(p, [0, 2, 1])], 2)  # [bn, 3, pn+3]
        W = tf.concat([W_0, W_1], 1)  # [bn, pn+3, pn+3]
        W_inv = tf.matrix_inverse(W)

        tp = tf.pad(coord + vector,
                    [[0, 0], [0, 3], [0, 0]], "CONSTANT")  # [bn, pn+3, 2]
        T = tf.matmul(W_inv, tp)  # [bn, pn+3, 2]
        T = tf.transpose(T, [0, 2, 1])  # [bn, 2, pn+3]

        return T

    T = _solve_system(coord, vector)
    y, x = _transform(T, coord, move, scal)
    input_transformed = _interpolate(U, y, x)
    output = tf.reshape(input_transformed, [num_batch, out_height, out_width, channels])
    y = tf.reshape(y, [num_batch, out_height, out_width, 1])
    x = tf.reshape(x, [num_batch, out_height, out_width, 1])
    t_arr = tf.concat([y, x], axis=-1)
    return output, t_arr

