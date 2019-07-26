import os
from dataloading import load_and_preprocess_image, dataset_map_train, dataset_map_test
from transformations import tps_parameters
from dotmap import DotMap
import numpy as np
from config import parse_args, write_hyperparameters
from model import Model
from utils import save_python_files, transformation_parameters, find_ckpt, batch_colour_map, save, initialize_uninitialized
import tensorflow as tf


def main(arg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu)
    model_save_dir = "../experiments/" + arg.name + "/"

    with tf.variable_scope("Data_prep"):
        if arg.mode == 'train':
            raw_dataset = dataset_map_train[arg.dataset](arg)

        elif arg.mode == 'predict':
            raw_dataset = dataset_map_test[arg.dataset](arg)

        dataset = raw_dataset.map(load_and_preprocess_image, num_parallel_calls=arg.data_parallel_calls)
        dataset = dataset.batch(arg['bn'], drop_remainder=True).repeat(arg.epochs)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        b_images = next_element

        orig_images = tf.tile(b_images, [2, 1, 1, 1])

        scal = tf.placeholder(dtype=tf.float32, shape=(), name='scal_placeholder')
        tps_scal = tf.placeholder(dtype=tf.float32, shape=(), name='tps_placeholder')
        rot_scal = tf.placeholder(dtype=tf.float32, shape=(), name='rot_scal_placeholder')
        off_scal = tf.placeholder(dtype=tf.float32, shape=(), name='off_scal_placeholder')
        scal_var = tf.placeholder(dtype=tf.float32, shape=(), name='scal_var_placeholder')
        augm_scal = tf.placeholder(dtype=tf.float32, shape=(), name='augm_scal_placeholder')

        tps_param_dic = tps_parameters(2 * arg.bn, scal, tps_scal, rot_scal, off_scal, scal_var)
        tps_param_dic.augm_scal = augm_scal

    ctr = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config=config) as sess:

        model = Model(orig_images, arg, tps_param_dic)
        tvar = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=tvar)
        merged = tf.summary.merge_all()

        if arg.mode == 'train':
            if arg.load:
                ckpt, ctr = find_ckpt(model_save_dir + 'saved_model/')
                saver.restore(sess, ckpt)
            else:
                save_python_files(save_dir=model_save_dir + 'bin/')
                write_hyperparameters(arg.toDict(), model_save_dir)
                sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter("../summaries/" + arg.name, graph=sess.graph)

        elif arg.mode == 'predict':
            ckpt, ctr = find_ckpt(model_save_dir + 'saved_model/')
            saver.restore(sess, ckpt)

        initialize_uninitialized(sess)
        while True:
            try:
                feed = transformation_parameters(arg, ctr, no_transform=(arg.mode == 'predict'))  # no transform if arg.visualize
                trf = {scal: feed.scal, tps_scal: feed.tps_scal,
                       scal_var: feed.scal_var, rot_scal: feed.rot_scal, off_scal: feed.off_scal, augm_scal: feed.augm_scal}
                ctr += 1
                if arg.mode == 'train':
                    if np.mod(ctr, arg.summary_interval) == 0:
                        merged_summary = sess.run(merged, feed_dict=trf)
                        writer.add_summary(merged_summary, global_step=ctr)

                    _, loss = sess.run([model.optimize, model.loss], feed_dict=trf)
                    if np.mod(ctr, arg.save_interval) == 0:
                        saver.save(sess, model_save_dir + '/saved_model/' + 'save_net.ckpt', global_step=ctr)

                elif arg.mode == 'predict':
                    img, img_rec, mu, heat_raw = sess.run([model.image_in, model.reconstruct_same_id, model.mu,
                                                           batch_colour_map(model.part_maps)], feed_dict=trf)

                    save(img, mu, ctr)

            except tf.errors.OutOfRangeError:
                print("End of training.")
                break


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    if arg.decoder == 'standard':
        if arg.reconstr_dim == 256:
            arg.rec_stages = [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
            arg.feat_slices = [[0, 0], [0, 0], [0, 0], [0, 0], [4, arg.n_parts], [2, 4], [0, 2]]
            arg.part_depths = [arg.n_parts, arg.n_parts, arg.n_parts, arg.n_parts, arg.n_parts, 4, 2]

        if arg.reconstr_dim == 128:
            arg.rec_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
            arg.feat_slices = [[0, 0], [0, 0], [0, 0], [4, arg.n_parts], [2, 4], [0, 2]]
            arg.part_depths = [arg.n_parts, arg.n_parts, arg.n_parts, arg.n_parts, 4, 2]
    main(arg)
