git README.md

Work in progress

Instructions:
Install tensorflow 1.14 

Download
a) video dataset or,
b) image dataset

Change dataloader
a) use the structure of load_train_human3m in dataloading.py 
b) use the structure of load_train_generic in dataloading.py

Start training (current hyperparameters work for roughly cropped human3m without background)
a) python main.py test_a --gpu 0 --dataset human3m --mode train --bn 16
b) python main.py test_b --gpu 0 --dataset generic --mode train --bn 16 --static 

Eval (use same flags as in training except for mode)
a) python main.py test_a --gpu 0 --dataset human3m --mode predict --bn 16
b )python main.py test_b --gpu 0 --dataset generic --mode train --bn 16 --static 


Todo:
-code documentation
-add preprocessing
-add hyperparameters for datasets
-add eval functions

[Project page with videos](https://compvis.github.io/unsupervised-disentangling/) 
