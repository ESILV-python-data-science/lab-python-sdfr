# -*- coding: utf-8 -*-
"""
Cluster images based on visual similarity

C. Kermorvant - 2017
"""


import argparse
import glob
import logging
import os
import shutil
import time
import sys

from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
import numpy as np



# default sub-resolution
IMG_FEATURE_SIZE = (12, 16)

# Setup logging
logger = logging.getLogger('cluster_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def extract_features(img):
    """
    Compute the subresolution of an image and return it as a feature vector

    :param img: the original image (can be color or gray)
    :type img: pillow image
    :return: pixel values of the image in subresolution
    :rtype: list of int in [0,255]

    """

    # convert color images to grey level
    gray_img = img.convert('L')
    # find the min dimension to rotate the image if needed
    min_size = min(img.size)
    if img.size[1] == min_size:
        # convert landscape  to portrait
        rotated_img = gray_img.rotate(90, expand=1)
    else:
        rotated_img = gray_img

    # reduce the image to a given size
    reduced_img = rotated_img.resize(
        IMG_FEATURE_SIZE, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]


def copy_to_dir(images, clusters, cluster_dir):
    """
    Move images to a directory according to their cluster name

    :param images: list of image names (path)
    :type images: list of path
    :param clusters: list of cluster values (int), such as given by cluster.labels_, associated to each image
    :type clusters: list
    :param cluster_dir: prefix path where to copy the images is a drectory corresponding to each cluster
    :type images: path
    :return: None
    """

    for img_path, cluster in zip(images, clusters):
        # define the cluster path : for example "CLUSTERS/4" if the image is in cluster 4
        clst_path = os.path.join(cluster_dir, str(cluster))
        # create the directory if it does not exists
        if not os.path.exists(clst_path):
            os.mkdir(clst_path)
        # copy the image into the cluster directory
        shutil.copy(img_path, clst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, cluster images and move them to a directory')
    parser.add_argument('--images-dir',required=True)
    parser.add_argument('--move-images')
    args = parser.parse_args()


    if args.move_images:
        CLUSTER_DIR = args.move_images
        # Clean up
        if os.path.exists(CLUSTER_DIR):
            shutil.rmtree(CLUSTER_DIR)
            logger.info('remove cluster directory %s' % CLUSTER_DIR)
        os.mkdir(CLUSTER_DIR)

    # find all the pages in the directory
    images_path_list = []
    data = []
    if args.images_dir:
        SOURCE_IMG_DIR = args.images_dir
        searchPath = os.path.join(args.images_dir, '*.jpg')
        images_path_list = glob.glob(searchPath)

    print(images_path_list[:])

    if not images_path_list:
        logger.warning("Did not found any jpg image in %s"%args.images_dir)
        sys.exit(0)

    # TODO : Extract the feature vector on all the pages found and store the feature
    # vectors in  data

    # cluster the feature vectors
    if not data:
        logger.error("Could not extract any feature vector")
        sys.exit(1)

    # convert to np array (default format for scikit-learn)
    X = np.array(data)
    logger.info("Running clustering")

    # TODO : run the K-Means clusering and call copy_to_dir to copy the image
    # in the directory corresponding to its cluster
