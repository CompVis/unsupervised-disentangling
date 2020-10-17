import os
from dataloading import (
    load_and_preprocess_image,
    dataset_map_train,
    dataset_map_test,
    keypoint_files_map,
)
from transformations import tps_parameters
from dotmap import DotMap
import numpy as np
from config import parse_args, write_hyperparameters
from model import Model
from utils import (
    save_python_files,
    save_part_transfer,
    transformation_parameters,
    find_ckpt,
    batch_colour_map,
    save_demo,
    save,
    save_no_kps,
    save_transfer,
    initialize_uninitialized,
    populate_demo_csv_file,
    merge_all_transfers
)
import tensorflow as tf
import numpy as np
from typing import *
import sklearn
import seaborn
from sklearn import *
import json
import cv2


def pck(distances, tolerance_pixels, image_size):
    # 6 pixel tolerance, normalized to 256 image distance
    pck = distances < (tolerance_pixels / image_size)
    return np.mean(pck)


def main(arg):
    populate_demo_csv_file("./datasets/deepfashion/images/demo/")
    model_save_dir = os.path.join("experiments", arg.name)
    with tf.variable_scope("Data_prep"):
        raw_dataset = dataset_map_test[arg.dataset](arg)

        dataset = raw_dataset.map(
            load_and_preprocess_image, num_parallel_calls=arg.data_parallel_calls
        )
        dataset = dataset.batch(arg["bn"], drop_remainder=True)
        iterator = dataset.make_initializable_iterator()
        # iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        b_images = next_element

        orig_images = tf.tile(b_images, [2, 1, 1, 1])

        scal = tf.placeholder(dtype=tf.float32, shape=(), name="scal_placeholder")
        tps_scal = tf.placeholder(dtype=tf.float32, shape=(), name="tps_placeholder")
        rot_scal = tf.placeholder(
            dtype=tf.float32, shape=(), name="rot_scal_placeholder"
        )
        off_scal = tf.placeholder(
            dtype=tf.float32, shape=(), name="off_scal_placeholder"
        )
        scal_var = tf.placeholder(
            dtype=tf.float32, shape=(), name="scal_var_placeholder"
        )
        augm_scal = tf.placeholder(
            dtype=tf.float32, shape=(), name="augm_scal_placeholder"
        )

        tps_param_dic = tps_parameters(
            2 * arg.bn, scal, tps_scal, rot_scal, off_scal, scal_var
        )
        tps_param_dic.augm_scal = augm_scal

    ctr = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    if "infer" in arg.mode:

        n = arg.part_idx
        filters_ = [[1] * n]
        for i in range(16):
            base_filter = [0,] * 16
            base_filter[i] = 1
            filters_.append(base_filter)
        filters_ += [[0] * n]

        with tf.Session(config=config) as sess:
            filter_var = tf.placeholder("float", (16,))
            model = Model(orig_images, arg, tps_param_dic, optimize=False, visualize=False, filter_ = filter_var)
            tvar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            saver = tf.train.Saver(var_list=tvar)
            merged = tf.summary.merge_all()

            ckpt, ctr = find_ckpt(os.path.join(model_save_dir, "saved_model"))
            base_ctr = ctr
            saver.restore(sess, ckpt)

            initialize_uninitialized(sess)
            mu_list = []

            for i, filter_ in enumerate(filters_):
                sess.run(iterator.initializer)

                while True:
                    try:
                        feed = transformation_parameters(
                            arg, ctr, no_transform=True
                        )  # no transform if arg.visualize
                        trf = {
                            scal: feed.scal,
                            tps_scal: feed.tps_scal,
                            scal_var: feed.scal_var,
                            rot_scal: feed.rot_scal,
                            off_scal: feed.off_scal,
                            augm_scal: feed.augm_scal,
                            model.filter_: np.array(filter_),
                        }
                        ctr += 1
                        img, img_rec, mu, heat_raw, part_maps_rgb  = sess.run(
                            [
                                model.image_in,
                                model.reconstruct_same_id,
                                model.mu,
                                batch_colour_map(model.part_maps),
                                model.color_maps
                            ],
                            feed_dict=trf
                        )

                        if not os.path.exists(os.path.join(model_save_dir,'images/')) or len(os.listdir(os.path.join(model_save_dir,'images/'))) < 8:
                                save_no_kps(img[: arg.bn, ...], ctr, model_save_dir, dirname="images")

                        save_no_kps(img_rec, ctr, model_save_dir, dirname="images_transfer")

                        part_maps_rgb /= part_maps_rgb.max(axis=(1,2), keepdims=True)

                        save_no_kps(part_maps_rgb, ctr, model_save_dir, dirname="part_maps_rgb")
                        save_part_transfer(img[: arg.bn, ...], img_rec, part_maps_rgb, ctr, model_save_dir, dirname="part_based_transfer_plots")

                        mu_list.append(mu[: arg.bn, ...])
                    except tf.errors.OutOfRangeError:
                        print("End of Prediction")
                        break
            merge_all_transfers(model_save_dir, read_dir="part_based_transfer_plots", write_dir="all_transfer_plots")


if __name__ == "__main__":
    arg = DotMap(vars(parse_args()))
    if arg.decoder == "standard":
        if arg.reconstr_dim == 256:
            arg.rec_stages = [
                [256, 256],
                [128, 128],
                [64, 64],
                [32, 32],
                [16, 16],
                [8, 8],
                [4, 4],
            ]
            arg.feat_slices = [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [4, arg.n_parts],
                [2, 4],
                [0, 2],
            ]
            arg.part_depths = [
                arg.n_parts,
                arg.n_parts,
                arg.n_parts,
                arg.n_parts,
                arg.n_parts,
                4,
                2,
            ]

        if arg.reconstr_dim == 128:
            arg.rec_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
            arg.feat_slices = [[0, 0], [0, 0], [0, 0], [4, arg.n_parts], [2, 4], [0, 2]]
            arg.part_depths = [arg.n_parts, arg.n_parts, arg.n_parts, arg.n_parts, 4, 2]
    main(arg)
