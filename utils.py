import functools
import tensorflow as tf
import numpy as np
import os
import matplotlib


matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import glob
from shutil import copyfile
from dotmap import DotMap




def wrappy(func):
    def wrapped(*args, **kwargs):
        with tf.variable_scope(func.__name__):
            return func(*args, **kwargs)

    return wrapped


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name):
                setattr(self, attribute, function(self, *args, **kwargs))
        return getattr(self, attribute)

    return decorator


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def probabilistic_switch(a, b, counter, scale=10000):
    """
    :param a:
    :param b:
    :param counter:
    :param scale: corresponds to decay  rate
    :return: at counter 0 a is returned with p=1. the probability decays
    asymptotically to p 0.5 with increasing counter values
    """
    p = counter / (2 * counter + scale)
    r = np.random.choice([a, b], p=[1 - p, p])
    return r


def evolve_a_to_b(min_max, time):
    if len(min_max) == 1:
        evolve = min_max[0]
    elif len(min_max) == 2:
        evolve = time * min_max[1] + (1 - time) * min_max[0]

    return evolve


def transformation_parameters(arg=None, ctr=None, no_transform=False):
    """
    if no transform: arg.scal is still used
    default for penn {'scal': 0.5, 'tps_scal': 0.05, 'rot_scal': 0.3, 'off_scal': 0.15, 'scal_var': 0.1}
    :param t:
    :param range:
    :return:
    """
    trf_arg = {}

    if no_transform:
        trf_arg['scal'] = arg.scal[0]
        trf_arg['tps_scal'] = 0.
        trf_arg['rot_scal'] = 0.
        trf_arg['off_scal'] = 0.
        trf_arg['scal_var'] = 0.
        trf_arg['augm_scal'] = 0.

    else:
        time = min(ctr / arg.schedule_scale, 1.)
        trf_arg['scal'] = evolve_a_to_b(arg.scal, time)
        trf_arg['tps_scal'] = evolve_a_to_b(arg.tps_scal, time)
        trf_arg['rot_scal'] = evolve_a_to_b(arg.rot_scal, time)
        trf_arg['off_scal'] = evolve_a_to_b(arg.off_scal, time)
        trf_arg['scal_var'] = evolve_a_to_b(arg.scal_var, time)
        trf_arg['augm_scal'] = evolve_a_to_b(arg.augm_scal, time)

    dotty = DotMap(trf_arg)
    return dotty


def batch_colour_map(heat_map):
    c = heat_map.get_shape().as_list()[-1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    colour = tf.constant(colour)
    colour_map = tf.einsum('bijk,kl->bijl', heat_map, colour)
    return colour_map


def np_batch_colour_map(heat_map):
    c = heat_map.shape[-1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    np_colour = np.array(colour)
    colour_map = np.einsum('bijk,kl->bijl', heat_map, np_colour)
    return colour_map


def identify_parts(image, raw, n_parts, version):
    image_base = np.array(Image.fromarray(image[0]).resize((64, 64))) / 255.
    base = image_base[:, :, 0] + image_base[ :, :, 1] + image_base[:, :, 2]
    directory = os.path.join('../images/' + str(version) + "/identify/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(n_parts):
        prlonint("hep")
        plt.imshow(raw[0, :, :, i] + 0.02 * base, cmap='gray')
        fname = directory + str(i) + '.png'
        plt.savefig(fname, bbox_inches='tight')


def save(img, mu, counter):
    batch_size, out_shape = img.shape[0], img.shape[1:3]
    marker_list = ["o", "v", "s", "|", "_"]
    directory = os.path.join('../images/landmarks/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    s = out_shape[0] // 8
    n_parts = mu.shape[-2]
    mu_img = (mu + 1.) / 2. * np.array(out_shape)[0]
    steps = batch_size
    step_size = 1

    for i in range(0, steps, step_size):
        plt.imshow(img[i])
        for j in range(n_parts):
            plt.scatter(mu_img[i, j, 1], mu_img[i, j, 0],  s=s, marker=marker_list[np.mod(j, len(marker_list))], color=cm.hsv(float(j / n_parts)))

        plt.axis('off')
        fname = directory + str(counter) + '_' + str(i) + '.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

@wrappy
def tf_summary_feat_and_parts(encoding_list, part_depths, visualize_features=False, square=True):
    for n, enc in enumerate(encoding_list):
        part_maps, feat_maps = enc[:, :, :, :part_depths[n]], enc[:, :, :, part_depths[n]:]
        if square:
            part_maps = part_maps ** 2
        color_part_map = batch_colour_map(part_maps)
        with tf.variable_scope("parts"):
            tf.summary.image(name="parts" + str(n), tensor=color_part_map, max_outputs=4)

        if visualize_features:
            if feat_maps.get_shape().as_list()[-1] > 0:
                with tf.variable_scope("feature_maps"):
                    if square:
                        feat_maps = feat_maps ** 2
                    color_feat_map = batch_colour_map(
                        feat_maps / tf.reduce_sum(feat_maps, axis=[1, 2], keepdims=True)) 
                    tf.summary.image(name="feat_maps" + str(n), tensor=color_feat_map ** 2, max_outputs=4)


@wrappy
def part_to_color_map(encoding_list, part_depths, size, square=True, ):
    part_maps = encoding_list[0][:, :, :, :part_depths[0]]
    if square:
        part_maps = part_maps ** 4
    color_part_map = batch_colour_map(part_maps)
    color_part_map = tf.image.resize_images(color_part_map, size=(size, size))

    return color_part_map



def save_python_files(save_dir):
    assert (not os.path.exists(save_dir))
    os.makedirs(save_dir)
    for file in glob.glob("*.py"):
        copyfile(src=file, dst=save_dir + file)


def find_ckpt(dir):
    filename = dir + 'checkpoint'
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.readline()
            ckpt = content.split('"')[1]
            print("found checkpoint :" + ckpt)
            print("counter set to", ckpt.split("-")[-1])
            return dir + ckpt, int(ckpt.split("-")[-1])
    else:
        raise FileNotFoundError


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)  
    return inp
