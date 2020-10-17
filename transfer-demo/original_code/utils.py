import functools
import tensorflow as tf
import numpy as np
import os
import matplotlib
from typing import *

matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import glob
from shutil import copyfile
from dotmap import DotMap
import cv2

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
    attribute = "_cache_" + function.__name__
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
    is_not_initialized = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars]
    )
    not_initialized_vars = [
        v for (v, f) in zip(global_vars, is_not_initialized) if not f
    ]

    print([str(i.name) for i in not_initialized_vars])  # only for testing
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
        trf_arg["scal"] = arg.scal[0]
        trf_arg["tps_scal"] = 0.0
        trf_arg["rot_scal"] = 0.0
        trf_arg["off_scal"] = 0.0
        trf_arg["scal_var"] = 0.0
        trf_arg["augm_scal"] = 0.0

    else:
        time = min(ctr / arg.schedule_scale, 1.0)
        trf_arg["scal"] = evolve_a_to_b(arg.scal, time)
        trf_arg["tps_scal"] = evolve_a_to_b(arg.tps_scal, time)
        trf_arg["rot_scal"] = evolve_a_to_b(arg.rot_scal, time)
        trf_arg["off_scal"] = evolve_a_to_b(arg.off_scal, time)
        trf_arg["scal_var"] = evolve_a_to_b(arg.scal_var, time)
        trf_arg["augm_scal"] = evolve_a_to_b(arg.augm_scal, time)

    dotty = DotMap(trf_arg)
    return dotty


def batch_colour_map(heat_map):
    c = heat_map.get_shape().as_list()[-1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    colour = tf.constant(colour)
    colour_map = tf.einsum("bijk,kl->bijl", heat_map, colour)
    return colour_map


def np_batch_colour_map(heat_map):
    c = heat_map.shape[-1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    np_colour = np.array(colour)
    colour_map = np.einsum("bijk,kl->bijl", heat_map, np_colour)
    return colour_map


def identify_parts(image, raw, n_parts, version):
    image_base = np.array(Image.fromarray(image[0]).resize((64, 64))) / 255.0
    base = image_base[:, :, 0] + image_base[:, :, 1] + image_base[:, :, 2]
    directory = os.path.join("../images/" + str(version) + "/identify/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(n_parts):
        prlonint("hep")
        plt.imshow(raw[0, :, :, i] + 0.02 * base, cmap="gray")
        fname = directory + str(i) + ".png"
        plt.savefig(fname, bbox_inches="tight")

def save_demo(img, mu, counter, model_dir, target_dir="demo-predictions"):
    batch_size, out_shape = img.shape[0], img.shape[1:3]
    marker_list = ["o", "v", "s", "|", "_"]
    directory = os.path.join(model_dir, target_dir)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    s = out_shape[0] // 8
    n_parts = mu.shape[-2]
    mu_img = mu
    steps = batch_size
    step_size = 1

    for i in range(0, steps, step_size):
      kp_coords = mu_img[i, :, ::-1]
      img_keypoints = draw_keypoint_markers(img[i], kp_coords)
      img_keypoints = cv2.cvtColor(img_keypoints, cv2.COLOR_RGB2BGR)
      img_keypoints = np.cast["uint8"](img_keypoints*255)

#        plt.imshow(img[i])
#        for j in range(n_parts):
#            plt.scatter(
#                mu_img[i, j, 1],
#                mu_img[i, j, 0],
#                s=s,
#                marker=marker_list[np.mod(j, len(marker_list))],
#                color=cm.hsv(float(j / n_parts)),
#            )

#        plt.axis("off")
        
      fname = os.path.join(directory, str(counter) + "_" + str(i) + ".png")
      print("Saving prediction " + fname)
      cv2.imwrite(fname, img_keypoints)

#        plt.savefig(fname, bbox_inches="tight")
#        plt.close()


def save(img, mu, counter, model_dir, dirname="images"):
    batch_size, out_shape = img.shape[0], img.shape[1:3]
    marker_list = ["o", "v", "s", "|", "_"]
    directory = os.path.join(model_dir, dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    s = out_shape[0] // 8
    n_parts = mu.shape[-2]
    mu_img = (mu + 1.0) / 2.0 * np.array(out_shape)[0]
    steps = batch_size
    step_size = 1

    for i in range(0, steps, step_size):
        plt.imshow(img[i])
        for j in range(n_parts):
            plt.scatter(
                mu_img[i, j, 1],
                mu_img[i, j, 0],
                s=s,
                marker=marker_list[np.mod(j, len(marker_list))],
                color=cm.hsv(float(j / n_parts)),
            )

        plt.axis("off")
        fname = os.path.join(directory, str(counter) + "_" + str(i) + ".png")
        print(fname)
        plt.savefig(fname, bbox_inches="tight")
        plt.close()

def save_transfer(imgs, imgs_transfer, model_dir, dirname="transfer_plots"):
    imgs_kps = imgs.copy()
    batch_size, x,y,z = imgs.shape
    directory = os.path.join(model_dir, dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    m = np.zeros((x,y,z), dtype="uint8")
    for i in range(batch_size):
        m = np.concatenate((m, imgs[i]), axis = 0)

    tn = imgs_transfer.shape[0]
    imgs_transfer = np.concatenate((imgs_transfer[tn//2:], imgs_transfer[:tn//2]))
    for i in range(batch_size):
        new_row = imgs_kps[i].copy()
        for j in range(i * batch_size, (i+1) * batch_size):
            new_row = np.concatenate((new_row, imgs_transfer[j]), axis = 0)
        m = np.concatenate((m, new_row), axis=1)
    fname = os.path.join(directory, "transfer_plot.png")
    m = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)
    m = np.cast["uint8"](m*255)
    print(fname)
    cv2.imwrite(fname, m)

def save_part_transfer(imgs, imgs_transfer, imgs_part_based, ctr, model_dir, dirname="transfer_plots"):
    imgs_kps = imgs.copy()
    batch_size, x,y,z = imgs.shape
    n_imgs_part_based, xpb, ypb,zpb = imgs_part_based.shape
    directory = os.path.join(model_dir, dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    m = np.zeros((x,y,z), dtype="uint8")
    for i in range(batch_size):
        m = np.concatenate((m, imgs[i]), axis = 0)

    tn = imgs_transfer.shape[0]
    imgs_transfer = np.concatenate((imgs_transfer[tn//2:], imgs_transfer[:tn//2]))
    for i in range(batch_size):
        new_row = imgs_kps[i].copy()
        for j in range(i * batch_size, (i+1) * batch_size):
            new_row = np.concatenate((new_row, imgs_transfer[j]), axis = 0)
        m = np.concatenate((m, new_row), axis=1)

    left_col = np.zeros((x,y,z), dtype="uint8")
    for i in range(n_imgs_part_based):
        img_resized = cv2.cvtColor(imgs_part_based[i], cv2.COLOR_RGB2BGR) 
        img_resized = cv2.resize(img_resized, dsize=(x,y), interpolation=cv2.INTER_CUBIC)
        img_resized = np.cast["uint8"](img_resized*255)
        left_col = np.vstack((left_col, img_resized))
    m = np.hstack((left_col, m))
    m = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)
    m = np.cast["uint8"](m*255)
    fname = os.path.join(directory, str(ctr) + ".png")
    print(fname)
    cv2.imwrite(fname, m)




def save_no_kps(img, counter, model_dir, dirname="images"):
    batch_size, out_shape = img.shape[0], img.shape[1:3]
    directory = os.path.join(model_dir, dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    s = out_shape[0] // 8
    steps = batch_size
    step_size = 1

    for i in range(0, steps, step_size):
        img_i = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR) * 255
        img_i = img_i.astype(np.uint8)
        fname = os.path.join(directory, str(counter) + "_" + str(i) + ".png")
        cv2.imwrite(fname, img_i)
        print(fname)

@wrappy
def tf_summary_feat_and_parts(
    encoding_list, part_depths, visualize_features=False, square=True
):
    for n, enc in enumerate(encoding_list):
        part_maps, feat_maps = (
            enc[:, :, :, : part_depths[n]],
            enc[:, :, :, part_depths[n] :],
        )
        if square:
            part_maps = part_maps ** 2
        color_part_map = batch_colour_map(part_maps)
        with tf.variable_scope("parts"):
            tf.summary.image(
                name="parts" + str(n), tensor=color_part_map, max_outputs=4
            )

        if visualize_features:
            if feat_maps.get_shape().as_list()[-1] > 0:
                with tf.variable_scope("feature_maps"):
                    if square:
                        feat_maps = feat_maps ** 2
                    color_feat_map = batch_colour_map(
                        feat_maps / tf.reduce_sum(feat_maps, axis=[1, 2], keepdims=True)
                    )
                    tf.summary.image(
                        name="feat_maps" + str(n),
                        tensor=color_feat_map ** 2,
                        max_outputs=4,
                    )


@wrappy
def part_to_color_map(
    encoding_list, part_depths, size, square=True,
):
    part_maps = encoding_list[0][:, :, :, : part_depths[0]]
    if square:
        part_maps = part_maps ** 4
    color_part_map = batch_colour_map(part_maps)
    color_part_map = tf.image.resize_images(color_part_map, size=(size, size))

    return color_part_map


def save_python_files(save_dir):
    assert not os.path.exists(save_dir)
    os.makedirs(save_dir)
    for file in glob.glob("*.py"):
        copyfile(src=file, dst=os.path.join(save_dir, file))


def find_ckpt(dir):
    filename = os.path.join(dir, "checkpoint")
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.readline()
            ckpt = content.split('"')[1]
            print("found checkpoint :" + ckpt)
            print("counter set to", ckpt.split("-")[-1])
            return os.path.join(dir, ckpt), int(ckpt.split("-")[-1])
    else:
        raise FileNotFoundError


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return inp


def populate_demo_csv_file(img_dir, csv_path="./datasets/deepfashion/data_demo.csv"):
    """ Populate the csv_path with image names from img_dir """
    images = [f for f in os.listdir(img_dir)]
    os.remove(csv_path)
    with open(csv_path, 'w') as f:
        f.write("id,filename\n")
        for (idx, img_path) in enumerate(images):
            f.write("" + str(idx) + "," + os.path.join("demo", img_path) + "\n")


def draw_keypoint_markers(
    img,
    keypoints,
    font_scale = 0.5,
    thickness = 2,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    marker_list=["o", "v", "x", "+", "<", "-", ">", "c"]
):
    """ Draw keypoints on image with markers
    Parameters
    ----------
    img : np.ndarray
        shaped [H, W, 3] array  in range [0, 1]
    keypoints : np.ndarray
        shaped [kp, 2] - array giving keypoint positions in range [-1, 1] for x and y.
        Keypoints[:, 0] is x-coordinate (horizontal).
    font_scale : int, optional
        openCV font scale passed to 'cv2.putText', by default 1
    thickness : int, optional
        openCV font thickness passed to 'cv2.putText', by default 2
    font : cv2.FONT_xxx, optional
        openCV font, by default cv2.FONT_HERSHEY_SIMPLEX
    Examples
    --------
        from skimage import data
        astronaut = data.astronaut()
        keypoints = np.stack([np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)], axis=1)
        img_marked = draw_keypoint_markers(astronaut,
                                            keypoints,
                                            font_scale=2,
                                            thickness=3)
        plt.imshow(img_marked)
    """

    img_marked = img.copy()
    keypoints = convert_range(keypoints, [-1, 1], [0, img.shape[0] - 1])
    colors = make_colors(
        keypoints.shape[0], bytes=False, cmap=plt.cm.inferno
    )
    for i, kp in enumerate(keypoints):
        text = marker_list[i % len(marker_list)]
        (label_width, label_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        textX = kp[0]
        textY = kp[1]
        font_color = colors[i]
        text_position = (
            textX - label_width / 2.0 - baseline,
            textY - label_height / 2.0 + baseline,
        )
        text_position = tuple([int(x) for x in text_position])
        img_marked = cv2.putText(
            img_marked,
            text,
            text_position,
            font,
            font_scale,
            font_color,
            thickness=thickness,
        )
    return img_marked


def convert_range(
    array, input_range, target_range
):
    """convert range of array from input range to target range
    Parameters
    ----------
    array: np.ndarray
        array in any shape
    input_range: Iterable[int]
        range of array values in format [min, max]
    output_range: Iterable[int]
        range of rescaled array values in format [min, max]
    Returns
    -------
    np.ndarray
        rescaled array
    Examples
    --------
        t = imageutils.convert_range(np.array([-1, 1]), [-1, 1], [0, 1])
        assert np.allclose(t, np.array([0, 1]))
        t = imageutils.convert_range(np.array([0, 1]), [0, 1], [-1, 1])
        assert np.allclose(t, np.array([-1, 1]))
    """
    if input_range[1] <= input_range[0]:
        raise ValueError
    if target_range[1] <= target_range[0]:
        raise ValueError

    a = input_range[0]
    b = input_range[1]
    c = target_range[0]
    d = target_range[1]
    return (array - a) / (b - a) * (d - c) + c



def make_colors(
    n_classes,
    cmap,
    bytes = False,
    with_background=False,
    background_color=np.array([1, 1, 1]),
    background_id=0,
):
    """make a color array using the specified colormap for `n_classes` classes
    # TODO: test new background functionality
    Parameters
    ----------
    n_classes: int
        how many classes there are in the mask
    cmap: Callable, optional, by default plt.cm.inferno
        matplotlib colormap handle
    bytes: bool, optional, by default False
        bytes option passed to `cmap`.
        Returns colors in range [0, 1] if False and range [0, 255] if True
    Returns
    -------
    colors: ndarray
        an array with shape [n_classes, 3] representing colors in the range [0, 1].
    """
    colors = cmap(np.linspace(0, 1, n_classes), alpha=False, bytes=bytes)[:, :3]
    if with_background:
        colors = np.insert(colors, background_id, background_color, axis=0)
    return colors


def merge_all_transfers(base_path, read_dir, write_dir):
    path = os.path.join(base_path, read_dir)
    save_dir = os.path.join(base_path, write_dir)
    imgs = [os.path.join(path, img) for img in sorted(os.listdir(path), key=lambda path: int(path.split(".")[0]))]
    first_filename = imgs[0]
    m = cv2.imread(first_filename)
    for img_path in imgs:
        transfer_plot = cv2.imread(img_path)
        m = np.vstack((m, transfer_plot))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + "/all_transfer_plots.png"
    print('All Transfer Plots: ', save_path)
    cv2.imwrite(save_path, m)

