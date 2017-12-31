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
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt


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

def train_and_test_classifier(clf,x_train, x_test, y_train, y_test):
    # Do Training@
    t0 = time.time()
    clf.fit(x_train, y_train)
    logger.info("Training  done in %0.3fs" % (time.time() - t0))

    # Do testing
    logger.info("Testing Classifier")
    t0 = time.time()
    predicted = clf.predict(x_test)

    # Print score produced by metrics.classification_report and return accuracy scores
    logger.info("Testing  done in %0.3fs" % (time.time() - t0))
    print(metrics.classification_report(y_test,predicted))
    train_accuracy_score = clf.score(x_train,y_train)
    test_accuracy_score = metrics.accuracy_score(y_test,predicted)

    return train_accuracy_score, test_accuracy_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save features in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--kNN',action='store_true', help='find best k for nearest_neighbors classifier')
    classifier_group.add_argument('--logistic-regression',action='store_true')
    classifier_group.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')
    curve_group = parser.add_mutually_exclusive_group(required=False)
    curve_group.add_argument('--learning-curve',action='store_true')
    curve_group.add_argument('--testing-curve',action='store_true')
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

    if args.limit_samples:
        df_input = df_input.sample(n=args.limit_samples)
        y = df_input['class']
        X = df_input.drop('class', axis = 1)


    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    # Train classifier
    logger.info("Training Classifier")


    if args.kNN:
        # Use train_test_split to create train/test/valid split
        X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, train_size=0.6)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.5)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Validation set size is {}".format(X_valid.shape))
        logger.info("Test set size is {}".format(X_test.shape))

        best_score = [1,0]

        # Train a kNN classifier with different values of k and report the train/valid/test accuracy.
        # Select the best value for k according to the accuracy on the dev set.

        for k in range(1,5):
            logger.info('Use kNN classifier with k= {}'.format(k))
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
            
            # Do Training
            t0 = time.time()
            clf.fit(X_train, y_train)
            logger.info("Training  done in %0.3fs" % (time.time() - t0))

            # Do testing
            logger.info("Testing Classifier")
            t0 = time.time()
            predicted = clf.predict(X_valid)

            # Print score produced by metrics.classification_report and metrics.accuracy_score
            logger.info("Testing  done in %0.3fs" % (time.time() - t0))
            score = accuracy_score(y_valid, predicted)
            print("Accuracy score = ", score)

            if score > best_score[1]:
                best_score = [k, score]

        # Report the performance of the classifier on the test set for the best value of k
        print()
        print("Best accracy score on the VALID set = ",  best_score[1], " / k = ", best_score[0])

        # Do testing and report he performance of the classifier for the best value of k based on the test set
        t0 = time.time()
        predicted = clf.predict(X_test)
        logger.info("Testing  done in %0.3fs" % (time.time() - t0))
        test_score = accuracy_score(y_test, predicted)
        print()
        print("Accuracy score on the TEST set  = ", test_score, " / k = ", best_score[0])
        sys.exit()

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8)
    logger.info("Train set size is {}".format(X_train.shape))
    logger.info("Test set size is {}".format(X_test.shape))

    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))
        clf = KNeighborsClassifier(n_neighbors=args.nearest_neighbors)

        # Do Training
        t0 = time.time()
        clf.fit(X_train, Y_train)
        logger.info("Training  done in %0.3fs" % (time.time() - t0))

        # Do testing
        logger.info("Testing Classifier")
        t0 = time.time()
        predicted = clf.predict(X_test)

        # Print score produced by metrics.classification_report and metrics.accuracy_score
        logger.info("Testing  done in %0.3fs" % (time.time() - t0))
        print(accuracy_score(Y_test, predicted))


    if args.logistic_regression:
        # create logistic regression classifier
        logger.info('Use logistic regression classifier')
        clf = LogisticRegression()
            
        # Do Training@
        t0 = time.time()
        clf.fit(X_train, Y_train)
        logger.info("Training  done in %0.3fs" % (time.time() - t0))

        # Do testing
        logger.info("Testing Classifier")
        t0 = time.time()
        predicted = clf.predict(X_test)

        # Print score produced by metrics.classification_report and metrics.accuracy_score
        logger.info("Testing  done in %0.3fs" % (time.time() - t0))
        score = accuracy_score(Y_test, predicted)
        print("Accuracy score = ", score)


    if args.learning_curve:
        # Study the impact of a growing training set on the accuracy
        # Report the training and test set accuracies for the 1NN, 2NN, kNN
        # (k being the best value for k previously found) and the Logisitic Regression.
        growing_training_set = [0.01,0.1,0.2,0.4,0.6,0.8,1.0]
        train_length=len(growing_training_set)
        train_sets=np.zeros(train_length)
        lr_accuracy_scores = np.zeros((train_length,2))
        onenn_accuracy_scores = np.zeros((train_length,2))
        twonn_accuracy_scores = np.zeros((train_length,2))
        knn_accuracy_scores = np.zeros((train_length,2))

        lr = LogisticRegression()
        onenn = neighbors.KNeighborsClassifier(n_neighbors=1)
        twonn = neighbors.KNeighborsClassifier(n_neighbors=2)
        knn = neighbors.KNeighborsClassifier(n_neighbors=3)

        for i in range(train_length):
            X_train_bis, X_test_bis, Y_train_bis, Y_test_bis = train_test_split(X_train,Y_train, train_size=growing_training_set[i], shuffle=False)
            logger.info("Train set size is {}% of original training set".format(growing_training_set[i]*100))
            train_sets[i]=len(Y_train_bis)

            # Logistic regression
            train_accuracy_score, test_accuracy_score = train_and_test_classifier(lr,X_train_bis,X_test,Y_train_bis,Y_test)
            lr_accuracy_scores[i][0]=train_accuracy_score
            lr_accuracy_scores[i][1]=test_accuracy_score

            # 1NN
            train_accuracy_score, test_accuracy_score = train_and_test_classifier(onenn,X_train_bis,X_test,Y_train_bis,Y_test)
            onenn_accuracy_scores[i][0]=train_accuracy_score
            onenn_accuracy_scores[i][1]=test_accuracy_score

            # 2NN
            train_accuracy_score, test_accuracy_score = train_and_test_classifier(twonn,X_train_bis,X_test,Y_train_bis,Y_test)
            twonn_accuracy_scores[i][0]=train_accuracy_score
            twonn_accuracy_scores[i][1]=test_accuracy_score

            # kNN
            train_accuracy_score, test_accuracy_score = train_and_test_classifier(knn,X_train_bis,X_test,Y_train_bis,Y_test)
            knn_accuracy_scores[i][0]=train_accuracy_score
            knn_accuracy_scores[i][1]=test_accuracy_score

        # Plot the training curves
        plt.figure()
        plt.title('1NN')
        plt.plot(train_sets, onenn_accuracy_scores)
        plt.figure()
        plt.title('2NN')
        plt.plot(train_sets, twonn_accuracy_scores)
        plt.figure()
        plt.title('3NN (best k)')
        plt.plot(train_sets, knn_accuracy_scores)
        plt.figure()
        plt.title('Logistic Regression')
        plt.plot(train_sets, lr_accuracy_scores)
        plt.show()

    if args.testing_curve:
        # Study the impact of a growing testing set on the accuracy
        growing_testing_set = [0.01,0.10,0.20,0.40,0.60,0.80,0.99]
        test_length=len(growing_testing_set)
        test_sets=np.zeros(test_length)
        mean_accuracy = np.zeros(test_length)
        std_accuracy = np.zeros(test_length)

        lr = LogisticRegression()
        knn = neighbors.KNeighborsClassifier(3)

        for i in range(test_length):
            test_accuracy = np.zeros(10)
            for j in range(10):
                X_train_bis, X_test_bis, Y_train_bis, Y_test_bis = train_test_split(X_test,Y_test, test_size=growing_testing_set[i])
                logger.info("Test set size is {}% of original testing set".format(growing_testing_set[i]*100))
                test_sets[i]=len(Y_test_bis)

                #train_accuracy_score, test_accuracy_score = train_and_test_classifier(lr,X_train,X_test_bis,Y_train,Y_test_bis)
                #test_accuracy[j]=test_accuracy_score

                train_accuracy_score, test_accuracy_score = train_and_test_classifier(knn,X_train,X_test_bis,Y_train,Y_test_bis)
                test_accuracy[j]=test_accuracy_score
            mean_accuracy[i]=np.mean(test_accuracy)
            std_accuracy[i]=np.std(test_accuracy)
            print(test_accuracy)


        print(mean_accuracy)
        print(std_accuracy)

        # Plot the testing curves (mean accuracy) on a plot with error bars (standard deviation of the accuracy)
        plt.figure()	
        plt.title('3NN')
        plt.errorbar(test_sets, mean_accuracy, yerr=std_accuracy,fmt='o')
        plt.show()

        
