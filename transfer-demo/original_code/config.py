import argparse
from dataloading import dataset_map_train
from architectures import decoder_map, encoder_map
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="name of the experiment")

    parser.add_argument("--part-idx", type=int, default=-1)

    # run setting
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "predict", "infer", "infer_eval", "eval"],
        required=True,
    )
    # parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--load", action="store_true")

    # dataset folder
    parser.add_argument("--dataset", choices=dataset_map_train.keys(), required=True)

    # options
    parser.add_argument("--covariance", action="store_true")
    parser.add_argument("--feat_shape", default=True, type=bool)
    parser.add_argument("--L1", action="store_true")
    parser.add_argument(
        "--pck_tolerance",
        type=int,
        default=6,
        help="tolerance of pck calculation in pixels",
    )

    parser.add_argument("--heat_feat_normalize", default=True, type=bool)
    parser.add_argument("--epochs", default=100000, type=int, help="number of epochs")

    # architectures
    parser.add_argument("--decoder", default="standard", choices=decoder_map.keys())
    parser.add_argument("--encoder", default="seperate", choices=encoder_map.keys())
    parser.add_argument(
        "--in_dim",
        default=256,
        type=int,
        choices=[128, 256],
        help="dim of input img 256 or 128",
    )
    parser.add_argument(
        "--reconstr_dim",
        default=256,
        type=int,
        choices=[128, 256],
        help="dim of reconstructed img 256 or 128",
    )
    parser.add_argument(
        "--heat_dim", default=64, type=int, choices=[64], help="dim of part_map (fixed)"
    )
    parser.add_argument(
        "--pad_size", default=25, type=int, help="input padding of images"
    )

    # modes
    parser.add_argument(
        "--l_2_scal",
        default=0.1,
        type=float,
        help="scale around part means that is considered for l2",
    )
    parser.add_argument("--l_2_threshold", default=0.2, type=float, help="")
    parser.add_argument("--L_inv_scal", default=0.8, type=float, help="")

    parser.add_argument(
        "--bn",
        default=32,
        type=int,
        help="batchsize if not slim and 2 * batchsize if slim",
    )
    parser.add_argument("--n_parts", default=16, type=int, help="number of parts")
    parser.add_argument(
        "--n_features", default=64, type=int, help="neurons of feature map layer"
    )
    parser.add_argument("--n_c", default=3, type=int)
    parser.add_argument(
        "--nFeat_1",
        default=256,
        type=int,
        help="neurons in residual module of part hourglass",
    )
    parser.add_argument(
        "--nFeat_2",
        default=256,
        type=int,
        help="neurons in residual module of feature hourglass",
    )

    # loss multiplication constants
    parser.add_argument(
        "--lr", default=0.001, type=float, help="learning rate of network"
    )
    parser.add_argument(
        "--lr_d",
        default=0.001,
        type=float,
        help="adversarial setting: learning rate  discriminator network",
    )

    parser.add_argument("--c_l2", default=1.0, type=float, help="")
    parser.add_argument("--c_trans", default=5.0, type=float, help="")
    parser.add_argument("--c_precision_trans", default=0.1, type=float, help="")
    parser.add_argument("--c_t", default=1.0, type=float, help="")

    # tps parameters
    parser.add_argument("--schedule_scale", default=100000, type=int, help="")
    parser.add_argument(
        "--scal",
        default=[0.8],
        type=float,
        nargs="+",
        help="default 0.6 sensible shedule [0.6, 0.6]",
    )
    parser.add_argument(
        "--tps_scal",
        default=[0.05],
        type=float,
        nargs="+",
        help="sensible shedule [0.01, 0.08]",
    )
    parser.add_argument(
        "--rot_scal",
        default=[0.1],
        type=float,
        nargs="+",
        help="sensible shedule [0.05, 0.6]",
    )
    parser.add_argument(
        "--off_scal",
        default=[0.15],
        type=float,
        nargs="+",
        help="sensible shedule [0.05, 0.15]",
    )
    parser.add_argument(
        "--scal_var",
        default=[0.05],
        type=float,
        nargs="+",
        help="sensible shedule [0.05, 0.2]",
    )
    parser.add_argument(
        "--augm_scal",
        default=[1.0],
        type=float,
        nargs="+",
        help="sensible shedule [0.0, 1.]",
    )

    # appearance parameters
    parser.add_argument(
        "--contrast_var", default=0.5, type=float, help="contrast variation"
    )
    parser.add_argument(
        "--brightness_var", default=0.3, type=float, help="contrast variation"
    )
    parser.add_argument(
        "--saturation_var", default=0.1, type=float, help="contrast variation"
    )
    parser.add_argument("--hue_var", default=0.3, type=float, help="contrast variation")
    parser.add_argument("--p_flip", default=0.0, type=float, help="contrast variation")

    # adverserial
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument(
        "--c_g",
        default=0.0002,
        type=float,
        help="factor weighting adversarial loss generator",
    )
    parser.add_argument(
        "--patch_size", default=[49, 49], type=int, nargs=2, help="dim of patch_size"
    )

    parser.add_argument("--print_vars", action="store_true")
    parser.add_argument(
        "--save_interval",
        default=20000,
        type=int,
        help="saves model every n gradient steps",
    )
    parser.add_argument(
        "--summary_interval",
        default=500,
        type=int,
        help="writes summary every n gradient steps",
    )

    parser.add_argument(
        "--static", action="store_true"
    )  # for e.g.birds (inter-species reconstruction too difficult)
    parser.add_argument(
        "--chunk_size",
        default=16,
        type=int,
        help="group of consecutive frames from video which are used for shape transformations",
    )
    parser.add_argument("--n_shuffle", default=64, type=int, help="n shuffle data")
    parser.add_argument(
        "--data_parallel_calls",
        default=4,
        type=int,
        help="number of parallel calls for tf map for preprocessing data",
    )
    parser.add_argument(
        "--num_steps",
        default=-1,
        type=int,
        help="optional number of stetps. -1 means run untitl manually terminating",
    )

    arg = parser.parse_args()
    return arg


def write_hyperparameters(r, save_dir):
    filename = os.path.join(save_dir, "config.txt")
    with open(filename, "a") as input_file:
        for k, v in r.items():
            line = "{}, {}".format(k, v)
            print(line)
            print >> input_file, line
            # print(line, file=input_file)
