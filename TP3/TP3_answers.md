# TP_classification

Select the best value for k according to the accuracy on the dev set. Report the performance performance of the classifier on the test set for this value of k.
> The best value of k on the valid set is 3: Accuracy score on the VALID set = 0.92775 / Accuracy score on the TEST set = 0.924666666667

Train a Logistic Regression classifier on 80% and test on 20% of the samples. Report the accuracy and compare to the best result of the kNN classifier.
> The best result of the kNN classifier (0.924666666667) is better than the result of the logistic regression classifier (0.859583333333)

Report the training and test set accuracies for the 1NN, 2NN, kNN (k being the best value for k you previously found) and the Logisitic Regresstion.

![img1](https://github.com/ESILV-python-data-science/lab-python-sdfr/blob/master/TP3/images/Figure_1.png?raw=true "1NN")

![img2](https://github.com/ESILV-python-data-science/lab-python-sdfr/blob/master/TP3/images/Figure_2.png?raw=true "2NN")

![img3](https://github.com/ESILV-python-data-science/lab-python-sdfr/blob/master/TP3/images/Figure_3.png?raw=true "3NN (best k)")

![img4](https://github.com/ESILV-python-data-science/lab-python-sdfr/blob/master/TP3/images/Figure_4.png?raw=true "logistic regression")

Report the mean and standard deviation (use np.mean and np.std) of the test set accuracy for the 1NN, 2NN, kNN (k being the best value for k you previously found) and the Logisitic Regression.


![img5](https://github.com/ESILV-python-data-science/lab-python-sdfr/blob/master/TP3/images/testing_1nn.png?raw=true "1NN")

![img6](https://github.com/ESILV-python-data-science/lab-python-sdfr/blob/master/TP3/images/testing_2nn.png?raw=true "2NN")

![img7](https://github.com/ESILV-python-data-science/lab-python-sdfr/blob/master/TP3/images/testing1.png?raw=true "3NN (best k)")

![img8](https://github.com/ESILV-python-data-science/lab-python-sdfr/blob/master/TP3/images/testing_log.png?raw=true "logistic regression")

How do you see that the estimation of the accuracy is more accurate when the test set size increases ?
> The more there are data, the lower the standard deviation is. This is how we know that de accuracy is more accurate.
