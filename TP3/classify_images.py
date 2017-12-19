# -*- coding: utf-8 -*-
"""
Classify digit images

C. Kermorvant - 2017
"""


import argparse
import logging
import time
import sys

from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn import svm, metrics, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import numpy as np


# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def extract_features_subresolution(img,img_feature_size = (8, 8)):
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
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save features in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')
    args = parser.parse_args()


    if args.load_features:
        df_input = pd.read_pickle(args.load_features)
        print(df_input.head())
        y = df_input['class']
        X = df_input.drop('class', axis = 1)
        # print(df_input)

    else:


        # Load the image list from CSV file using pd.read_csv
        # see the doc for the option since there is no header ;
        # specify the column names :  filename , class
        df = pd.read_csv('MNIST_all.csv', names=['filename', 'class'])
        file_list = []
        class_list = []
        current_row = 0
        file_list = df.filename

        # Show results
        #print(file_list)
        logger.info('Loaded {} images in {}'.format(df.shape,args.images_list))

        # Extract the feature vector on all the pages found
        # Modify the extract_features from TP_Clustering to extract 8x8 subresolution values
        # white must be 0 and black 255
        data = []
        for i_path in tqdm(file_list):
            page_image = Image.open(i_path)
            data.append(extract_features_subresolution(page_image))

        # check that we have data
        if not data:
            logger.error("Could not extract any feature vector or class")
            sys.exit(1)



        # convert to np.array
        X = np.array(data)
        y = df['class']
        logger.info("Running clustering")



    # save features
    if args.save_features:
        # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle
        df_features = pd.DataFrame(X)
        df_features['class'] = y
        df_features.to_pickle(args.save_features)
        #logger.info('Saved {} features and class to {}'.format(df_features.shape,args.save_features))


    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    # Train classifier
    logger.info("Training Classifier")

    # Use train_test_split to create train/test split
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, train_size=0.6)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.5)
    logger.info("Train set size is {}".format(X_train.shape))
    logger.info("Validation set size is {}".format(X_valid.shape))
    logger.info("Test set size is {}".format(X_test.shape))

    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))
        clf = KNeighborsClassifier(n_neighbors=args.nearest_neighbors)
    else:
        logger.error('No classifier specified')
        sys.exit()

    # Do Training@
    t0 = time.time()
    clf.fit(X_train, y_train)
    logger.info("Training  done in %0.3fs" % (time.time() - t0))

    # Do testing
    logger.info("Testing Classifier")
    t0 = time.time()
    predicted = clf.predict(X_valid)

    # Print score produced by metrics.classification_report and metrics.accuracy_score
    logger.info("Testing  done in %0.3fs" % (time.time() - t0))
    print(accuracy_score(y_valid, predicted))

    # After testing different values of k, we find that the best value is 6. Let's print the score on the test set

    # Do testing
    logger.info("Testing Classifier")
    t0 = time.time()
    predicted = clf.predict(X_test)

    # Print score produced by metrics.classification_report and metrics.accuracy_score
    logger.info("Testing  done in %0.3fs" % (time.time() - t0))
    print(accuracy_score(y_test, predicted))
    
