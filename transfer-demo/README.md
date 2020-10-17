
Checkout the Google Colab Demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rDNxZH1BYlyvBmE1z30Dng2TgUKnH3MP)


To setup the project locally:

1. Create a conda environment using the conda.env file as base

```bash
conda create --name <name> --file conda.env
```

This will install all the necessary dependencies for model demonstration

2. After installing the dependencies download the deepfashion checkpoint from [here](https://heibox.uni-heidelberg.de/f/c2e7b6a77f2f4736a01f/?dl=1) ando move it to the `original_code/experiments/baseline_deepfashion_256/` like this:



```bash
cd original_code/
rm -rf baseline_deepfashion_256*
wget --quiet https://heibox.uni-heidelberg.de/f/c2e7b6a77f2f4736a01f/?dl=1 -O baseline_deepfashion_256_checkpoints.tgz
tar -xvf baseline_deepfashion_256_checkpoints.tgz
mkdir -p -v experiments/baseline_deepfashion_256/demo-predictions
cp -r -v -t experiments/baseline_deepfashion_256 release/*
rm -rf release/
```

3. Run the model using the run script

```bash
cd original_code/
./run.sh
```

4. Checkout the stacked transfer plots saved in `experiments/baseline_deepfashion_256/all_transfer_plots/all_transfer_plots.png`. Individual transfer plots are in `experiments/baseline_deepfashion_256/part_based_transfer_plots/**`


5. [Project page with videos](https://compvis.github.io/unsupervised-disentangling/) 

6. Google Colab Live Demo Again [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rDNxZH1BYlyvBmE1z30Dng2TgUKnH3MP)
