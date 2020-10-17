import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import numpy as np


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        elem = l[i : i + n]
        random.shuffle(elem)
        yield elem


def load_train_human3m(arg, path="../datasets/human3/train/"):
    """
    creates tf.dataset from the human3m video dataset with the following folder structure on disk:
    human3
        train
            S1
                directions <- first video
                    0.jpg
                    1.jpg
                    2.jpg
                    ...
                discussion <- second video
                    0.jpg
                    1.jpg
                    2.jpg
                    ...
                ...
            S5
                directions
                    0.jpg
                    1.jpg
                    2.jpg
                    ...
                discussion
                    0.jpg
                    1.jpg
                    2.jpg
                    ...
                ...
            ...
    :param path:
    :return:
    """
    vids = [f for f in glob.glob(path + "*/*", recursive=True)]
    frames = []
    for vid in vids:
        for chunk in chunks(
            sorted(
                glob.glob(vid + "/*.jpg", recursive=True),
                key=lambda x: int(x.split("/")[-1].split(".jpg")[0]),
            ),
            arg.chunk_size,
        ):
            if len(chunk) == arg.chunk_size:
                random.shuffle(chunk)
                frames.append(chunk)
    random.shuffle(frames)
    frames = np.asarray(frames)
    raw_dataset = tf.data.Dataset.from_tensor_slices(frames).interleave(
        lambda x: tf.data.Dataset.from_tensor_slices(x).shuffle(
            arg.n_shuffle, reshuffle_each_iteration=True
        ),
        cycle_length=arg.chunk_size,
        block_length=2,
    )
    return raw_dataset


def load_test_human3m(arg, path="../datasets/human3/test/"):
    chunk_size = 2
    vids = [f for f in glob.glob(path + "*/*", recursive=True)]
    frames = []
    for vid in vids:
        for chunk in chunks(
            sorted(
                glob.glob(vid + "/*.jpg", recursive=True),
                key=lambda x: int(x.split("/")[-1].split(".jpg")[0]),
            ),
            chunk_size,
        ):
            if len(chunk) == 2:
                frames.append(chunk)
    frames = np.asarray(frames)
    raw_dataset = tf.data.Dataset.from_tensor_slices(frames).flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x)
    )
    return raw_dataset


def load_train_generic(arg, path="../datasets/generic/train_images/"):
    frames = glob.glob(path + "*.jpg", recursive=True)
    frames = np.asarray(frames).reshape(-1, 1)
    raw_dataset = (
        tf.data.Dataset.from_tensor_slices(frames)
        .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        .shuffle(arg.n_shuffle, reshuffle_each_iteration=True)
    )
    return raw_dataset


def load_test_generic(arg, path="../datasets/generic/test_images/"):
    frames = glob.glob(path + "*.jpg", recursive=True)
    frames = np.asarray(frames).reshape(-1, 1)
    raw_dataset = tf.data.Dataset.from_tensor_slices(frames).flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x)
    )
    return raw_dataset


def load_train_from_csv_nonstatic(
    arg,
    data_root="datasets/deepfashion/images",
    data_csv="datasets/deepfashion/data_train.csv",
    id_col_name="id",
    fname_col_name="filename",
):
    import pandas as pd

    data_frame = pd.read_csv(data_csv)
    cid_groups = data_frame.groupby(id_col_name)
    frames = []
    for _, group in cid_groups:
        abs_paths = group[fname_col_name].apply(lambda x: os.path.join(data_root, x))
        abs_paths = sorted(abs_paths)
        for chunk in chunks(abs_paths, arg.chunk_size):
            if len(chunk) == arg.chunk_size:
                random.shuffle(chunk)
                frames.append(chunk)
    random.shuffle(frames)
    frames = np.asarray(frames)
    raw_dataset = tf.data.Dataset.from_tensor_slices(frames).interleave(
        lambda x: tf.data.Dataset.from_tensor_slices(x).shuffle(
            arg.n_shuffle, reshuffle_each_iteration=True
        ),
        cycle_length=arg.chunk_size,
        block_length=2,
    )
    return raw_dataset


def load_test_from_csv_nonstatic(
    arg,
    data_root="datasets/deepfashion/images",
    data_csv="datasets/deepfashion/data_test.csv",
    id_col_name="id",
    fname_col_name="filename",
):
    import pandas as pd

    data_frame = pd.read_csv(data_csv)
    cid_groups = data_frame.groupby(id_col_name)
    frames = []
    for _, group in cid_groups:
        abs_paths = group[fname_col_name].apply(lambda x: os.path.join(data_root, x))
        abs_paths = sorted(abs_paths)
        for chunk in chunks(abs_paths, arg.chunk_size):
            if len(chunk) == arg.chunk_size:
                frames.append(chunk)
    frames = np.asarray(frames)
    raw_dataset = tf.data.Dataset.from_tensor_slices(frames).flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x)
    )
    return raw_dataset


def load_train_from_csv_static(
    arg,
    data_root="datasets/deepfashion/images",
    data_csv="datasets/deepfashion/data_test.csv",
    id_col_name="id",
    fname_col_name="filename",
):
    # frames = glob.glob(path + "*.jpg", recursive=True)
    import pandas as pd

    data_frame = pd.read_csv(data_csv)
    abs_paths = data_frame[fname_col_name].apply(lambda x: os.path.join(data_root, x))
    frames = np.asarray(abs_paths).reshape(-1, 1)
    raw_dataset = (
        tf.data.Dataset.from_tensor_slices(frames)
        .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        .shuffle(arg.n_shuffle, reshuffle_each_iteration=True)
    )
    return raw_dataset


def load_test_from_csv_static(
    arg,
    data_root="datasets/deepfashion/images",
    data_csv="datasets/deepfashion/data_demo.csv",
    id_col_name="id",
    fname_col_name="filename",
):
    # frames = glob.glob(path + "*.jpg", recursive=True)
    import pandas as pd

    data_frame = pd.read_csv(data_csv)
    abs_paths = data_frame[fname_col_name].apply(lambda x: os.path.join(data_root, x))
    frames = np.asarray(abs_paths).reshape(-1, 1)
    raw_dataset = tf.data.Dataset.from_tensor_slices(frames).flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x)
    )
    return raw_dataset


import functools

data_root = "datasets/deepfashion/images"
data_csv = "datasets/deepfashion/data_train.csv"
load_train_deepfashion = functools.partial(
    load_train_from_csv_static,
    data_root=data_root,
    data_csv=data_csv,
    id_col_name="id",
    fname_col_name="filename",
)

data_csv = "datasets/deepfashion/data_test.csv"
load_test_deepfashion = functools.partial(
    load_test_from_csv_static,
    data_root=data_root,
    data_csv=data_csv,
    id_col_name="id",
    fname_col_name="filename",
)

dataset_map_train = {
    "generic": load_train_generic,
    "human3m": load_train_human3m,
    "csv": load_train_from_csv_nonstatic,
    "deepfashion": load_train_from_csv_static,
}

dataset_map_test = {
    "generic": load_test_generic,
    "human3m": load_test_human3m,
    "csv": load_test_from_csv_nonstatic,
    "deepfashion": load_test_from_csv_static,
}


keypoint_files_map = {"deepfashion": "datasets/deepfashion/data_test.json"}

