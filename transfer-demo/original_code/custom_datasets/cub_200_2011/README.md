# Cub Datasets used for experiments in Lorenz et al., UNSUPERVISED PART-BASED DISENTANGLING OF OBJECT SHAPE AND APPEARANCE



## Setting up the data

1. Download the `images` folder

```
https://heibox.uni-heidelberg.de/f/1e3f582936a74cfb8861/?dl=1

```

2. unpack it

```
pigz -dc cub_loren19_images.tar.gz | pv | tar xf -
mv cub_loren19_images images
```


## Dataset format

* `train.csv` and `test.csv` contain lists with columns `idx`, `fname`, `fname_original`, `category`
    * `idx` is just an enumeration of the file index
    * `fname` is the relative path from this directory on
    * `fname_original` is the filename from the original cub_200_2011 dataset
    * `category` is the category index from the cub_200_2011 dataset

* `train_kp.npy` and `test_kp.npy` arrays shaped `N, 15, 3`.
    * The 15 means that thare are 15 keypoints
    * The 3 represents coordinates x, y and binary visiblity (visibility == 1 means visible)
    * Coordinates are in range `[0, 1]`, so to get to pixel coordinates one has to scale them with the image size

## Additional notes

* The images are **not** the same as in the original cub_200_2011 dataset.
* They are cropped and aligned so that the viewing direction is to the left and the eye keypoint is top left. 
* The image size is *300x300*, however I did not test if this is the case for all images.