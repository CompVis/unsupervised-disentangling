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
    transformation_parameters,
    find_ckpt,
    batch_colour_map,
    save_demo,
    save,
    save_no_kps,
    save_transfer,
    initialize_uninitialized,
    populate_demo_csv_file
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
        iterator = dataset.make_one_shot_iterator()
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
        with tf.Session(config=config) as sess:

            model = Model(orig_images, arg, tps_param_dic, optimize=False, visualize=False)
            tvar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            saver = tf.train.Saver(var_list=tvar)
            merged = tf.summary.merge_all()

            ckpt, ctr = find_ckpt(os.path.join(model_save_dir, "saved_model"))
            saver.restore(sess, ckpt)

            initialize_uninitialized(sess)
            mu_list = []
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
                    }
                    ctr += 1

                    img, img_rec, mu, heat_raw = sess.run(
                        [
                            model.image_in,
                            model.reconstruct_same_id,
                            model.mu,
                            batch_colour_map(model.part_maps),
                        ],
                        feed_dict=trf,
                    )
                    save_no_kps(img[: arg.bn, ...], ctr, model_save_dir, dirname="images")
                    save_no_kps(img_rec, ctr, model_save_dir, dirname="images_transfer")
                    save(img[: arg.bn, ...], mu[:arg.bn, ...], ctr, model_save_dir, dirname="images_kps")
                    save_transfer(img[:arg.bn, ...], img[:arg.bn, ], img_rec, model_save_dir, dirname="transfer_plots")
                    mu_list.append(mu[: arg.bn, ...])
                except tf.errors.OutOfRangeError:
                    print("End of Prediction")
                    break
            print("Saving outputs")
            mu_list = np.concatenate(mu_list, axis=0)
            np.savez_compressed(
                os.path.join(model_save_dir, "keypoints_predicted.npz"),
                keypoints_predicted=mu_list,
            )

    if "eval" in arg.mode:
        # TODO: extract method
        if arg.dataset in ["deepfashion"]:
            # regress to keypoints
            with np.load(
                os.path.join(model_save_dir, "keypoints_predicted.npz")
            ) as data:
                keypoints_predicted = data["keypoints_predicted"]

            with open(keypoint_files_map[arg.dataset], "rb",) as f:
                gt_keypoint_data = json.load(f)
                gt_keypoints = (
                    np.stack([d["keypoints"] for d in gt_keypoint_data], axis=0) / 256.0
                )
                joint_order = gt_keypoint_data[0]["joint_order"]

            N = keypoints_predicted.shape[0]

            X_train = keypoints_predicted[:N, ...].reshape(N, -1)
            y_train = gt_keypoints[:N, ...].reshape(N, -1)
            X_test = X_train

            regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=False)
            _ = regr.fit(X_train, y_train)
            y_predict = regr.predict(X_test)
            regressed_keypoints = y_predict.reshape(N, -1, 2)
            joint_order = gt_keypoint_data[0]["joint_order"]
            landmarks_gt = gt_keypoints[:N, ...]
            landmarks_regressed = y_predict.reshape((N, -1, 2))
            distances = np.linalg.norm(landmarks_gt - landmarks_regressed, axis=-1)
            np.savez_compressed(
                os.path.join(model_save_dir, "keypoints_regressed.npz"),
                regressed_keypoints=landmarks_regressed,
                distances=distances,
            )

            import seaborn
            from matplotlib import pyplot as plt
            import pandas as pd

            table = pd.DataFrame(distances, columns=joint_order.values())

            plt.style.use("seaborn-whitegrid")
            NB_RC_PARAMS = {
                "figure.figsize": [8, 5],
                "figure.dpi": 220,
                "figure.autolayout": True,
                "legend.frameon": True,
            }
            with plt.rc_context(NB_RC_PARAMS):
                ax = table.boxplot(rot=45, showfliers=False, fontsize=12)
                ax.set_ylabel(r"$||e||$")
                ax.set_ylim([0, 0.1])
                plt.savefig(os.path.join(model_save_dir, "keypoint_distances.png"))

            pck_value = pck(distances, arg.pck_tolerance, arg.in_dim)
            with open(os.path.join(model_save_dir, "metrics.txt"), "w") as f:
                print("pck : {:.0f}%".format(100 * pck_value))


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
