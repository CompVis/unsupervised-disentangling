#! /usr/bin/env bash

# python 3.6
# pip install tensorflow==1.14
# pip install dotmap 
# conda install matplotlib
# conda install pillow
# conda install typing
# conda install opencv
# conda install dataclasses
# conda install -c conda-forge pudb
# conda install scikit-learn
# conda install seaborn
# conda install -c conda-forge gdown

# python "$(dirname "$0")/transfer-demo.py" baseline_deepfashion_256 \
#     --dataset deepfashion \
#     --bn 8 \
#     --static \
#     --in_dim 256 \
#     --reconstr_dim 256 \
#     --covariance \
#     --scal 1.0 \
#     --contrast_var 0.01 \
#     --brightness_var 0.01 \
#     --saturation_var 0.01 \
#     --hue_var 0.01 \
#     --adversarial \
#     --mode infer \
#     --pad_size 50 \
#     --pck_tolerance 6 2>&1 | grep -Evi 'warning'
PART_IDX=16
if [ -n "$1" ]; then
    PART_IDX=$1
fi


python "$(dirname "$0")/transfer-parts-demo-fast.py" baseline_deepfashion_256 \
    --dataset deepfashion \
    --bn 8 \
    --static \
    --in_dim 256 \
    --reconstr_dim 256 \
    --covariance \
    --scal 1.0 \
    --contrast_var 0.01 \
    --brightness_var 0.01 \
    --saturation_var 0.01 \
    --hue_var 0.01 \
    --adversarial \
    --mode infer \
    --pad_size 0 \
    --part-idx "$PART_IDX"  \
    --pck_tolerance 6 2>&1 | grep -Evi 'warning'
