from PIL import Image
import numpy as np
import os
import unittest

def get_image_as_array(base_path, img_name):
    return np.asarray(Image.open(base_path + "/" + img_name), dtype="uint8")


def build_upper_row(matrix, n, upper_row_path):
    mx, my, mz = matrix.shape
    x, y = (mx//(n+1), my//(n+1))
    paths = os.listdir(upper_row_path)
    ctr = 2
    for i, path in enumerate(paths):
        img = get_image_as_array(upper_row_path, path)
        matrix[(x*(i+1)):(x*ctr), 0:y, ...] = img
        ctr += 1

def build_left_col(matrix, n, left_col_path):
    mx, my, mz = matrix.shape
    x, y = (mx//(n+1), my//(n+1))
    paths = os.listdir(left_col_path)
    ctr = 2
    for i, path in enumerate(paths):
        img = get_image_as_array(left_col_path, path)
        matrix[0:x, (y*(i+1)):(y*ctr), ...] = img
        ctr += 1

def print_array(arr):
    img = Image.fromarray(arr)
    img.show()

def build_transfer(matrix, n, transfer_path):
    mx, my, mz = matrix.shape
    x, y = (mx//(n+1), my//(n+1))
    paths = sorted(os.listdir(transfer_path))
    ctr = 0
    print(paths)
    for i, path in enumerate(paths):
        if i % n == 0: ctr += 1
        img = get_image_as_array(transfer_path, path)
        x1 = x*(((i + 1) % n)+1)
        x2 = x*(((i + 1) % n)+2)
        matrix[x1:x2, (y*ctr):(y*(ctr+1)), ...] = img

def save_transfer(m, path):
    print(m.shape)
    img = Image.fromarray(m)
    img.show()
    img.save(path + 'transfer_matrix.png', format='PNG')

def generate_transfer_matrix(upper_row_path, left_col_path, transfer_path):
    """Returns a transfer matrix given three paths to upper row images, 
    left side images and transfer images.
    It assumes all images are the same width and height"""
    paths = os.listdir(upper_row_path)
    n = len(paths)
    x,y,z = get_image_as_array(upper_row_path, paths[0]).shape
    mx, my = ((n+1) * x, (n+1) * y)
    m = np.zeros((mx, my, z), dtype="uint8")
    build_upper_row(m, n, upper_row_path)
    build_left_col(m, n, left_col_path)
    build_transfer(m, n, transfer_path)
    save_transfer(m, './')
    return m

transfer_matrix = generate_transfer_matrix('./experiments/baseline_deepfashion_256/images',
    './experiments/baseline_deepfashion_256/images_kps',
    './experiments/baseline_deepfashion_256/images_transfer')

# X############### <- input images (upper row)
# ################ <- transfer images
# ################
# ################
# ################
# ################
# ################
# ################
# ################
# ################
# ################
# ################
# ################
# ################
# ################
# ################

# ^
# |
# KPS images
