
import gdown
url = 'https://drive.google.com/uc?id=1H2KLseDjs28Yi7EIAXf3jwIeVBtKTY1l'
output = '../datasets/deepfashion/img.zip'
gdown.download(url, output, quiet=False)
