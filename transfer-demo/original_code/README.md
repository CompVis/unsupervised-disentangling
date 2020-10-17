# Inofficial steps to reproduce the results

* I tried many ways to reproduce the results and found that the following parameters work well.


**IF YOU FIND SOMETHING INCONSISTENT, PLEASE MAKE AN ISSUE**. 
Srsly, please make an issue so that I can fix it. Thanks :blush: :thumbsup:


## Deepfashion

* Deepfashion is trained on single images, so it is a *static* dataset.


### Training the baseline
```bash
export CUDA_VISIBLE_DEVICES=X # instead of main.py --gpu xx
python main.py baseline_deepfashion_256 \
--dataset deepfashion --mode train --bn 8 --static \
--in_dim 256 \
--reconstr_dim 256 \
--covariance \
--pad_size 25 \
--contrast_var 0.01 \
--brightness_var 0.01 \
--saturation_var 0.01 \
--hue_var 0.01 \
--adversarial \
--c_precision_trans 0.01


export CUDA_VISIBLE_DEVICES=X # instead of main.py --gpu xx
python main.py baseline_deepfashion_256_nonadversarial \
--dataset deepfashion --mode train --bn 8 --static \
--in_dim 256 \
--reconstr_dim 256 \
--covariance \
--pad_size 25 \
--contrast_var 0.01 \
--brightness_var 0.01 \
--saturation_var 0.01 \
--hue_var 0.01 \
--c_precision_trans 0.01 \
--num_steps 500001
```

* Note that I had to make a custom split of the data for Deepfashion, which is basically going through all the data in the 
in-shop subset of Deepfashion and filter out those images where all keypoints are visible.
* To get the keypoints, I simply used Alpha Pose.
* The custom subset is released under [custom_datasets/deepfashion](custom_datasets/deepfashion/README.md)


A pretrained checkpoint is available [here](https://heibox.uni-heidelberg.de/f/c2e7b6a77f2f4736a01f/?dl=1).



### Inferring keypoints

```bash
python predict.py baseline_deepfashion_256 \
--dataset deepfashion --bn 16 --static \
--in_dim 256 \
--reconstr_dim 256 \
--covariance \
--contrast_var 0.01 \
--brightness_var 0.01 \
--saturation_var 0.01 \
--hue_var 0.01 \
--adversarial \
--mode infer_eval \
--pck_tolerance 6
```


```bash
python predict.py baseline_deepfashion_256_nonadversarial \
--dataset deepfashion --bn 16 --static \
--in_dim 256 \
--reconstr_dim 256 \
--covariance \
--contrast_var 0.01 \
--brightness_var 0.01 \
--saturation_var 0.01 \
--hue_var 0.01 \
--mode infer_eval \
--pck_tolerance 6
```


|     | Adversarial | Non-adversarial |
| --- | ----------- | -------------- |
| PCK | 57%         | 52%            |

* this means that the adversarial is probably important