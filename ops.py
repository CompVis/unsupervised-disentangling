import tensorflow as tf
import numpy as np
from utils import wrappy
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


def AbsDetJacobian(batch_meshgrid):
    """
    :param batch_meshgrid: takes meshgrid tensor of dim [bn, h, w, 2] (conceptually meshgrid represents a two dimensional function f = [fx, fy] on [bn, h, w] )
    :return: returns Abs det of  Jacobian of f of dim [bn, h, w, 1 ]
    """
    y_c = tf.expand_dims(batch_meshgrid[:, :, :, 0], -1)
    x_c = tf.expand_dims(batch_meshgrid[:, :, :, 1], -1)
    sobel_x = 1 / 4 * tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    filtered_y_y = tf.nn.conv2d(y_c, sobel_y_filter, strides=[1, 1, 1, 1], padding='VALID')
    filtered_y_x = tf.nn.conv2d(y_c, sobel_x_filter, strides=[1, 1, 1, 1], padding='VALID')
    filtered_x_y = tf.nn.conv2d(x_c, sobel_y_filter, strides=[1, 1, 1, 1], padding='VALID')
    filtered_x_x = tf.nn.conv2d(x_c, sobel_x_filter, strides=[1, 1, 1, 1], padding='VALID')

    Det = tf.abs(filtered_y_y * filtered_x_x - filtered_y_x * filtered_x_y)
    pad = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    Det = tf.pad(Det, pad, mode='SYMMETRIC')

    return Det


@wrappy
def augm(t, arg):
    t = tf.image.random_contrast(t, lower=1 - arg.contrast_var, upper=1 + arg.contrast_var)
    t = tf.image.random_brightness(t, arg.brightness_var)
    t = tf.image.random_saturation(t, 1 - arg.saturation_var, 1 + arg.saturation_var)
    t = tf.image.random_hue(t, max_delta=arg.hue_var)

    random_tensor = 1. - arg.p_flip + random_ops.random_uniform(shape=[1], dtype=t.dtype)

    binary_tensor = math_ops.floor(random_tensor)
    augmented = binary_tensor * t + (1 - binary_tensor) * (1 - t)
    return augmented

@wrappy
def Parity(t_images, t_mesh, on=False):
    if on:
        bn = t_images.get_shape().as_list()[0]
        P = tf.random_uniform(shape=[bn, 1, 1, 1], dtype=tf.float32) - 0.5  # bn, h ,w ,c
        P = tf.cast(P > 0., dtype=tf.float32)
        Pt_images = P * t_images[:, :, ::-1] + (1 - P) * t_images
        Pt_mesh = P * t_mesh[:, :, ::-1] + (1 - P) * t_mesh

    else:
        Pt_images = t_images
        Pt_mesh = t_mesh

    return Pt_images, Pt_mesh



@wrappy
def prepare_pairs(t_images, reconstr_dim, arg):
    if arg.mode == 'train':
        bn, h, w, n_c = t_images.get_shape().as_list()
        if arg.static:
            t_images = tf.concat([tf.expand_dims(t_images[:bn//2], axis=1), tf.expand_dims(t_images[bn//2:], axis=1)], axis=1)
        else:
            t_images = tf.reshape(t_images, shape=[bn // 2, 2, h, w, n_c])
        t_c_1_images = tf.map_fn(lambda x: augm(x, arg), t_images)
        t_c_2_images = tf.map_fn(lambda x: augm(x, arg), t_images)
        a, b = tf.expand_dims(t_c_1_images[:, 0], axis=1), tf.expand_dims(t_c_1_images[:, 1], axis=1)
        c, d = tf.expand_dims(t_c_2_images[:, 0], axis=1), tf.expand_dims(t_c_2_images[:, 1], axis=1)
        if arg.static:
            t_input_images = tf.reshape(tf.concat([a, d], axis=0), shape=[bn, h, w, n_c])
            t_reconstr_images = tf.reshape(tf.concat([c, b], axis=0), shape=[bn, h, w, n_c])
        else:
            t_input_images = tf.reshape(tf.concat([a, d], axis=1), shape=[bn, h, w, n_c])
            t_reconstr_images = tf.reshape(tf.concat([c, b], axis=1), shape=[bn, h, w, n_c])

        t_input_images = tf.clip_by_value(t_input_images, 0., 1.)
        t_reconstr_images = tf.image.resize_images(tf.clip_by_value(t_reconstr_images, 0., 1.), size=(reconstr_dim, reconstr_dim))

    else:
        t_input_images = tf.clip_by_value(t_images, 0., 1.)
        t_reconstr_images = tf.image.resize_images(tf.clip_by_value(t_images, 0., 1.), size=(reconstr_dim, reconstr_dim))

    return t_input_images, t_reconstr_images


@wrappy
def reverse_batch(tensor, n_reverse):
    """
    reverses order of elements the first axis of tensor
    example: reverse_batch(tensor=tf([[1],[2],[3],[4],[5],[6]), n_reverse=3) returns tf([[3],[2],[1],[6],[5],[4]]) for n reverse 3
    :param tensor:
    :param n_reverse:
    :return:
    """
    bn, *rest = tensor.get_shape().as_list()
    assert ((bn / n_reverse).is_integer())
    tensor = tf.reshape(tensor, shape=[bn // n_reverse, n_reverse, *rest])
    tensor_rev = tensor[:, ::-1]
    tensor_rev = tf.reshape(tensor_rev, shape=[bn, *rest])
    return tensor_rev


@wrappy
def softmax_norm(logit_map):
    eps = 1e-12
    exp = tf.exp(logit_map - tf.reduce_max(logit_map, axis=[1, 2], keepdims=True))
    norm = tf.reduce_sum(exp, axis=[1, 2], keepdims=True) + eps
    softmax = exp / norm
    return softmax, norm


@wrappy
def softmax(logit_map):
    eps = 1e-12
    exp = tf.exp(logit_map - tf.reduce_max(logit_map, axis=[1, 2], keepdims=True))
    norm = tf.reduce_sum(exp, axis=[1, 2], keepdims=True) + eps
    softmax = exp / norm
    return softmax



def random_scal(bn, min_scal, max_scal):
    rand_scal = tf.random_uniform(shape=[bn // 2, 2],
                                  minval=min_scal, maxval=max_scal, dtype=tf.float32)
    rand_scal = tf.tile(rand_scal, [2, 2])
    rand_scal = tf.reshape(rand_scal, shape=[2 * bn, 2])
    return rand_scal



@wrappy
def part_map_to_mu_L_inv(part_maps, scal):
    """
    Calculate mean for each channel of part_maps
    :param part_maps: tensor of part map activations [bn, h, w, n_part]
    :return: mean calculated on a grid of scale [-1, 1]
    """
    bn, h, w, nk = part_maps.get_shape().as_list()
    y_t = tf.tile(tf.reshape(tf.linspace(-1., 1., h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1., 1., w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)

    mu = tf.einsum('ijl,aijk->akl', meshgrid, part_maps)
    mu_out_prod = tf.einsum('akm,akn->akmn', mu, mu)

    mesh_out_prod = tf.einsum('ijm,ijn->ijmn', meshgrid, meshgrid)
    stddev = tf.einsum('ijmn,aijk->akmn', mesh_out_prod, part_maps) - mu_out_prod

    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-12

    a = tf.sqrt(a_sq + eps)  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = tf.sqrt(b_sq_add_c_sq - b ** 2 + eps)
    z = tf.zeros_like(a)

    tf.summary.scalar(name="L_0_0", tensor=a[0, 0])
    tf.summary.scalar(name="L_1_0", tensor=b[0, 0])
    tf.summary.scalar(name="L_1_1", tensor=c[0, 0])

    det = tf.expand_dims(tf.expand_dims(a * c, axis=-1), axis=-1)
    row_1 = tf.expand_dims(tf.concat([tf.expand_dims(c, axis=-1), tf.expand_dims(z, axis=-1)], axis=-1), axis=-2)
    row_2 = tf.expand_dims(tf.concat([tf.expand_dims(-b, axis=-1), tf.expand_dims(a, axis=-1)], axis=-1), axis=-2)

    L_inv = scal / (det + eps) * tf.concat([row_1, row_2], axis=-2)  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
    tf.summary.scalar(name="L_inv_0_0", tensor=L_inv[0, 0, 0, 0])
    tf.summary.scalar(name="L_inv_1_0", tensor=L_inv[0, 0, 1, 0])
    tf.summary.scalar(name="L_inv_1_1", tensor=L_inv[0, 0, 1, 1])

    return mu, L_inv


@wrappy
def get_features(features, part_map, slim):
    """
    :param features: features of shape [bn, h, w, n_features] (slim), or [bn, h, w, n_part, n_features]
    :param part_map:  part_map of shape [bn, h, w, n_part]
    :param slim:
    :return: features of shape [bn, nk, n_features]
    """
    if slim:
        features = tf.einsum('bijf,bijk->bkf', features, part_map)
    else:
        features = tf.einsum('bijkf,bijk->bkf', features, part_map)
    return features


@wrappy
def augm_mu(image_in, image_rec, mu, features, batch_size, n_part, move_list):
    image_in = tf.tile(tf.expand_dims(image_in[0], axis=0),  [batch_size, 1, 1, 1])
    image_rec = tf.tile(tf.expand_dims(image_rec[0], axis=0),  [batch_size, 1, 1, 1])
    mu = tf.tile(tf.expand_dims(mu[0], axis=0), [batch_size, 1, 1])
    features = tf.tile(tf.expand_dims(features[0], axis=0), [batch_size, 1, 1])
    batch_size = batch_size // 2
    ran = (tf.reshape(tf.range(batch_size), [batch_size, 1]))/batch_size - 0.5
    array = tf.concat([tf.concat([ran, tf.zeros_like(ran)], axis=-1), tf.concat([tf.zeros_like(ran), ran], axis=-1)], axis=0)
    array = tf.expand_dims(tf.cast(array, dtype=tf.float32), axis=1)
    for elem in move_list:
        part = tf.constant(elem, dtype=tf.int32, shape=([1, 1]))
        pad_part = tf.constant([[1, 1], [0, 0]])
        part_arr = tf.pad(tf.concat([part, -part], axis=-1), pad_part)
        pad = tf.constant([[0, 0], [0, n_part - 1], [0, 0]]) + part_arr
        addy = tf.pad(array, pad)
        mu = mu + addy
    return image_in, image_rec, mu, features


@wrappy
def precision_dist_op(precision, dist, part_depth, nk, h, w):
    proj_precision = tf.einsum('bnik,bnkf->bnif', precision, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = tf.reduce_sum(proj_precision, axis=-2)  # sum x and y axis

    heat = 1 / (1 + proj_precision)
    heat = tf.reshape(heat, shape=[-1, nk, h, w])  # bn width height number parts
    part_heat = heat[:, :part_depth]
    part_heat = tf.transpose(part_heat, [0, 2, 3, 1])
    return heat, part_heat


@wrappy
def feat_mu_to_enc(features, mu, L_inv, reconstruct_stages, part_depths, feat_map_depths, static, n_reverse, covariance=None, feat_shape=None, heat_feat_normalize=True, range=10, ):
    """
    :param features: tensor shape   bn, nk, nf
    :param mu: tensor shape  [bn, nk, 2] in range[-1,1]
    :param L_inv: tensor shape  [bn, nk, 2, 2]
    :param reconstruct_stages:
    :param part_depths:
    :param feat_map_depths:
    :param n_reverse:
    :param average:
    :return:
    """
    bn, nk, nf = features.get_shape().as_list()

    if static:
        reverse_features = tf.concat([features[bn//2:], features[:bn//2]], axis=0)

    else:
        reverse_features = reverse_batch(features, n_reverse)

    encoding_list = []
    circular_precision = tf.tile(tf.reshape(tf.constant([[range, 0.], [0, range]], dtype=tf.float32),  shape=[1, 1, 2, 2]), multiples=[bn, nk, 1, 1])

    for dims, part_depth, feat_slice in zip(reconstruct_stages, part_depths, feat_map_depths):
        h, w = dims[0], dims[1]

        y_t = tf.expand_dims(tf.tile(tf.reshape(tf.linspace(-1., 1., h), [h, 1]), [1, w]), axis=-1)
        x_t = tf.expand_dims(tf.tile(tf.reshape(tf.linspace(- 1., 1., w), [1, w]), [h, 1]), axis=-1)

        y_t_flat = tf.reshape(y_t, (1, 1, 1, -1))
        x_t_flat = tf.reshape(x_t, (1, 1, 1, -1))

        mesh = tf.concat([y_t_flat, x_t_flat], axis=-2)
        dist = mesh - tf.expand_dims(mu, axis=-1)

        if not covariance or not feat_shape:
            heat_circ, part_heat_circ = precision_dist_op(circular_precision, dist, part_depth, nk, h, w)

        if covariance or feat_shape:
            heat_shape, part_heat_shape = precision_dist_op(L_inv, dist, part_depth, nk, h, w)

        nkf = feat_slice[1] - feat_slice[0]

        if nkf != 0:
            feature_slice_rev = reverse_features[:, feat_slice[0]: feat_slice[1]]

            if feat_shape:
                heat_scal = heat_shape[:, feat_slice[0]: feat_slice[1]]

            else:
                heat_scal = heat_circ[:, feat_slice[0]: feat_slice[1]]

            if heat_feat_normalize:
                heat_scal_norm = tf.reduce_sum(heat_scal, axis=1, keepdims=True) + 1
                heat_scal = heat_scal / heat_scal_norm

            heat_feat_map = tf.einsum('bkij,bkn->bijn', heat_scal, feature_slice_rev)

            if covariance:
                encoding_list.append(tf.concat([part_heat_shape, heat_feat_map], axis=-1))

            else:
                encoding_list.append(tf.concat([part_heat_circ, heat_feat_map], axis=-1))

        else:
            if covariance:
                encoding_list.append(part_heat_shape)

            else:
                encoding_list.append(part_heat_circ)

    return encoding_list


@wrappy
def heat_map_function(y_dist, x_dist, y_scale, x_scale):
    x = 1 / (1 + (tf.square(y_dist / (1e-6 + y_scale)) + tf.square(
        x_dist / (1e-6 + x_scale))))
    return x


@wrappy
def unary_mat(vector):
    b_1 = tf.expand_dims(vector, axis=-2)  # (y, x) #b1 and b2 are eigenvectors
    b_2 = tf.expand_dims(tf.einsum('bkc,c->bkc', vector[:, :, ::-1], tf.constant([-1., 1], dtype=tf.float32)),
                         axis=-2)  # (y, x) -> (-x, y) get  orthogonal eigvec
    U_mat = tf.concat([b_1, b_2], axis=-2)  # U = (b_1^T, b_2^t)^T U contains transposed eigenvecs
    return U_mat


@wrappy
def get_img_slice_around_mu(img, mu, slice_size):
    """

    :param img:
    :param mu: in range [-1, 1]
    :param slice_size:
    :return: bn, n_part, slice_size[0] , slice_size[1], channel colour + n_part
    """

    h, w = slice_size
    bn, img_h, img_w, c = img.get_shape().as_list()  # bn this actually 2bn now
    bn_2, nk, _ = mu.get_shape().as_list()
    assert (int(h / 2))
    assert (int(w / 2))
    assert (bn_2 == bn)

    scal = tf.constant([img_h, img_w], dtype=tf.float32)
    mu = tf.stop_gradient(mu)
    mu_no_grad = tf.einsum('bkj,j->bkj', (mu + 1) / 2., scal)
    mu_no_grad = tf.cast(mu_no_grad, dtype=tf.int32)

    mu_no_grad = tf.reshape(mu_no_grad, shape=[bn, nk, 1, 1, 2])
    y = tf.tile(tf.reshape(tf.range(- h // 2, h // 2), [1, 1, h, 1, 1]), [bn, nk, 1, w, 1])
    x = tf.tile(tf.reshape(tf.range(- w // 2, w // 2), [1, 1, 1, w, 1]), [bn, nk, h, 1, 1])

    field = tf.concat([y, x], axis=-1) + mu_no_grad

    h1 = tf.tile(tf.reshape(tf.range(bn), [bn, 1, 1, 1, 1]), [1, nk, h, w, 1])

    idx = tf.concat([h1, field], axis=-1)

    image_slices = tf.gather_nd(img, idx)
    return image_slices




@wrappy
def fold_img_with_mu(img, mu, scale, visualize, threshold, normalize=True):
    """
    folds the pixel values of img with potentials centered around the part means (mu)
    :param img: batch of images
    :param mu:  batch of part means in range [-1, 1]
    :param scale: scale that governs the range of the potential
    :param visualize:
    :param normalize: whether to normalize the potentials
    :return: folded image
    """
    bn, h, w, nc = img.get_shape().as_list()
    bn, nk, _ = mu.get_shape().as_list()

    py = tf.expand_dims(mu[:, :, 0], 2)
    px = tf.expand_dims(mu[:, :, 1], 2)

    py = tf.stop_gradient(py)
    px = tf.stop_gradient(px)

    y_t = tf.tile(tf.reshape(tf.linspace(-1., 1., h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(- 1., 1., w), [1, w]), [h, 1])
    x_t_flat = tf.reshape(x_t, (1, 1, -1))
    y_t_flat = tf.reshape(y_t, (1, 1, -1))

    y_dist = py - y_t_flat
    x_dist = px - x_t_flat

    heat_scal = heat_map_function(y_dist=y_dist, x_dist=x_dist, x_scale=scale, y_scale=scale)
    heat_scal = tf.reshape(heat_scal, shape=[bn, nk, h, w])  # bn width height number parts
    heat_scal = tf.einsum('bkij->bij', heat_scal)
    heat_scal = tf.clip_by_value(t=heat_scal, clip_value_min=0., clip_value_max=1.)
    heat_scal = tf.where(heat_scal > threshold, heat_scal, tf.zeros_like(heat_scal))

    norm = tf.reduce_sum(heat_scal, axis=[1, 2], keepdims=True)
    tf.summary.scalar("norm", tensor=tf.reduce_mean(norm))
    if normalize:
        heat_scal_norm = heat_scal / norm
        folded_img = tf.einsum('bijc,bij->bijc', img, heat_scal_norm)
    if not normalize:
        folded_img = tf.einsum('bijc,bij->bijc', img, heat_scal)
    if visualize:
        tf.summary.image(name="foldy_map", tensor=tf.expand_dims(heat_scal, axis=-1), max_outputs=4)

    return folded_img, tf.expand_dims(heat_scal, axis=-1)




@wrappy
def mu_img_gate(mu, resolution, scale):
    """
    folds the pixel values of img with potentials centered around the part means (mu)
    :param img: batch of images
    :param mu:  batch of part means in range [-1, 1]
    :param scale: scale that governs the range of the potential
    :param visualize:
    :param normalize: whether to normalize the potentials
    :return: folded image
    """
    bn, nk, _ = mu.get_shape().as_list()

    py = tf.expand_dims(mu[:, :, 0], 2)
    px = tf.expand_dims(mu[:, :, 1], 2)

    py = tf.stop_gradient(py)
    px = tf.stop_gradient(px)
    h, w = resolution

    y_t = tf.tile(tf.reshape(tf.linspace(-1., 1., h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(- 1., 1., w), [1, w]), [h, 1])
    x_t_flat = tf.reshape(x_t, (1, 1, -1))
    y_t_flat = tf.reshape(y_t, (1, 1, -1))

    y_dist = py - y_t_flat
    x_dist = px - x_t_flat

    heat_scal = heat_map_function(y_dist=y_dist, x_dist=x_dist, x_scale=scale, y_scale=scale)
    heat_scal = tf.reshape(heat_scal, shape=[bn, nk, h, w])  # bn width height number parts
    heat_scal = tf.einsum('bkij->bij', heat_scal)
    return heat_scal

@wrappy
def binary_activation(x):
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out


@wrappy
def fold_img_with_L_inv(img, mu, L_inv, scale, visualize, threshold, normalize=True):
    """
    folds the pixel values of img with potentials centered around the part means (mu)
    :param img: batch of images
    :param mu:  batch of part means in range [-1, 1]
    :param scale: scale that governs the range of the potential
    :param visualize:
    :param normalize: whether to normalize the potentials
    :return: folded image
    """
    bn, h, w, nc = img.get_shape().as_list()
    bn, nk, _ = mu.get_shape().as_list()

    mu_stop = tf.stop_gradient(mu)

    y_t = tf.tile(tf.reshape(tf.linspace(-1., 1., h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(- 1., 1., w), [1, w]), [h, 1])
    x_t_flat = tf.reshape(x_t, (1, 1, -1))
    y_t_flat = tf.reshape(y_t, (1, 1, -1))

    mesh = tf.concat([y_t_flat, x_t_flat], axis=-2)
    dist = mesh - tf.expand_dims(mu_stop, axis=-1)

    proj_precision = tf.einsum('bnik,bnkf->bnif', scale * L_inv, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = tf.reduce_sum(proj_precision, axis=-2)  # sum x and y axis

    heat = 1 / (1 + proj_precision)

    heat = tf.reshape(heat, shape=[bn, nk, h, w])  # bn width height number parts
    heat = tf.einsum('bkij->bij', heat)
    heat_scal = tf.clip_by_value(t=heat, clip_value_min=0., clip_value_max=1.)
    heat_scal = tf.where(heat_scal > threshold, heat_scal, tf.zeros_like(heat_scal))
    norm = tf.reduce_sum(heat_scal, axis=[1, 2], keepdims=True)
    tf.summary.scalar("norm", tensor=tf.reduce_mean(norm))
    if normalize:
        heat_scal = heat_scal / norm
    folded_img = tf.einsum('bijc,bij->bijc', img, heat_scal)
    if visualize:
        tf.summary.image(name="foldy_map", tensor=tf.expand_dims(heat_scal, axis=-1), max_outputs=4)

    return folded_img


@wrappy
def probabilistic_switch(handles, handle_probs, counter, scale=10000.):
    t = counter / scale
    scheduled_probs = []

    for p_1, p_2 in zip(handle_probs[::2], handle_probs[1::2]):
        scheduled_prob = t * p_1 + (1 - t) * p_2
        scheduled_probs.append(scheduled_prob)

    handle = np.random.choice(handles, p=scheduled_probs)
    return handle


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars])  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))



