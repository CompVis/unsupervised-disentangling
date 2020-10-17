* This is a subset of all person images from the **Deepfashion inshop** datasets where all keypoints are visible

* To get this list, we simply use the heuristic that keypoint confidences should be above 0.5 for all joints in the image.
* To get the keypoints, we used the Alpha Pose keypoint estimation model.

* To use the csv files, symlink the deepfashion inshop dataset here

* symlink the data from deepfashion_inshop here

```bash
ln -s -d xxx/deepfashion_inshop/Img/img images
```

* The final folder structure should look something like this
```
datasets/deepfashion
│   data_train.csv
│   data_test.csv 
│
└───images
    |   
    └───MEN
    │   │   DENIM
    │   │   Jackets_Vests
    |   |   ...
    |
    └───WOMEN
        │   Blouses_Shirts
        │   Cardigans
        | ...
```