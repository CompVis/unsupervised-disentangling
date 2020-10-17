#! /usr/bin/env bash

N_KEYPOINTS=16
IMGS_PATH="experiments/baseline_deepfashion_256"

mkdir -p "$IMGS_PATH/part_based_transfer_plots"

for IDX in $(seq -1 $((N_KEYPOINTS - 1))); do
    ./run.sh $IDX
    cp -v -f "$IMGS_PATH/transfer_plots/transfer_plot.png" "$IMGS_PATH/part_based_transfer_plots/transfer_plot_$IDX.png"
done

if [ ! -f experiments/baseline_deepfashion_256/all_transfer_plots/all_transfer_plots.png ]; then 
    python ./merge-transfer-plots.py $N_KEYPOINTS
fi
