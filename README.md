# Digit_Classifier
Multiple Linear Regressor which classifies written digits. PCA feature extraction and cross-validation are used to train the algorithm.

## Python setup

1.	The “Project Interpreter” used is a 64 bits version of Python 2.7.
2.	The code was written on a 64 bits version of PyCharm Community Edition 2016.
3.	The following packages were installed and used for the homework through ”pip tool installer” version 9.0.1:

<center>
Package | Version
  ---   |   ---
numpy   | 1.11.2+mkl
scipy   | 0.18.1
sklearn | 0.18.1
</center>
4.	The “basicDemo.m” file was executed in MATLAB. Then the Workspace was saved as “basicDemo.mat”.
5.	“basicDemo.mat” was stored in the same folder as the python project directory.

## Cross-validation scheme

The classifier performance is evaluated through k-fold cross-validation. This cross-validation method divides the training data set into k folds or subsets of the same size. Then, the classifier is trained using k-1 folds and evaluated with the remaining fold. This process is repeated k times, each time training and evaluating the classifier using different folds. In this way, the classifier is evaluated with data that has never been seen during training.

When dividing the training data set into the k folds some care must be taken. All the classes must be represented so that it is ensured that the classifier has seen all the classes during training. Additionally, it is desirable that the ratio of examples in each class in the original training set is maintained in the folds.

I choose 5 folds. The cross-validation performance is reported as the average of the performance in the 5 iterations.

## Feature selection

The approach to extract features is using PCA in the entire training set. The parameter to choose is the number of principal components. Using many components has the risk of overfitting (in addition to providing more data to the classification algorithm, which may slow down the training process). For instance, using as many as 200 principal components the cross-validation (CV) accuracy obtained is 92% and the test misclassification rate of 8%. Lowering the number of principal components to 25 gives a slightly larger CV accuracy of 93% and lower misclassification rate in the test set: 6%.
