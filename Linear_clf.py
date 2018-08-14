########################################################################################################################
# File_name:            Linear_clf-Moises_Garcia.py                                                                    #
# Student:              Moises Daniel Garcia Rojas                                                                     #
# Created:              Friday - December 2nd, 2016                                                                    #
# Last modification:    Friday - December 9th, 2016                                                                    #
# Description:          Training a feature-based linear classifier for the digits dataset "mfeat-pix.txt".             #
#                       The "basicDemo.m" file was used to generate the MATLAB workspace "basicDemo.mat"               #
#                       and used to train the linear classifier through cross-validation to optimize feature           #
#                       repertoire.                                                                                    #
#                       The "basicDemo.m" and "mfeat-pix.txt" files were provided at class                             #
########################################################################################################################

import numpy as np
import scipy.io
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# The classifier was created based on LinearRegression as given in basicDemo.m
# The classifier follows sklearn hierarchy (inheriting from BaseEstimator and
# ClassifierMixin) and implements the methods fit and predict. In this way I
# can make use of sklearn's cross-validation.
class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        # Check that the data have the correct shape. This means that X and y have consistent length
        X, y = check_X_y(X, y)

        # Store the classes seen during training.
        self.classes_ = unique_labels(y)

        # Create labels matrix. E.g.: if y is [0,0,1,2] the matrix will be
        # [[1,1,0,0], [0,0,1,0], [0,0,0,1]].
        # +1 because we assume classes to be represented from 0 to 9.
        num_classes = np.max(y) + 1
        # Check all classes are represented in training.
        assert (np.all(self.classes_ == np.arange(num_classes)))
        Y = np.zeros((num_classes, len(y)))
        Y[y, np.arange(len(y))] = 1

        # Train the linear regression model.
        self.regr_ = type('LinearRegression', (object,), {'coef_': None})

        # Linear regression training as in basicDemo.m, see W variable.
        self.regr_.coef_ = np.matmul(np.linalg.inv(np.matmul(X.T, X)),
                                     np.matmul(X.T, Y.T)).T

        return self

    def predict(self, X):
        # Check that fit has been previously called.
        check_is_fitted(self, ['classes_'])

        X = check_array(X)
        # Classification as in basicDemo.m.
        return np.argmax(np.matmul(self.regr_.coef_, X.T), axis=0)


# Load Matlab workspace data to employ training and test sets created with the basicDemo.m.
basicDemoData = scipy.io.loadmat('basicDemo.mat', squeeze_me=True)
# Get correctLabels from basicDemo.mat.
correctLabels = basicDemoData['correctLabels']
# Transform from 1-indexing in Matlab to 0-indexing in Python.
correctLabels -= 1

# First stage - feature extraction.
# Estimation of PCA features from all training data.

# 25 principal components were selected. Using more components (e.g. 100 or more) leads to
# overfitting and the performance in the test set decreases.
num_pca_components = 25
pca = PCA(n_components=num_pca_components)


print('number of principal components = %d' % num_pca_components)
pca.fit(basicDemoData['trainPatterns'])
# Get feature values from train and test sets.
featureValuesTrain = pca.transform(basicDemoData['trainPatterns'])
featureValuesTest = pca.transform(basicDemoData['testPatterns'])

# Second stage - learn classifier and validate in the training set.
clf = LinearRegressionClassifier()
scores = cross_val_score(clf, featureValuesTrain, correctLabels, cv=5)
print("cross-validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Third stage - evaluate classifier in test set.
clf.fit(featureValuesTrain, correctLabels)
print('test misclassification rate = %0.2f' % (1 - clf.score(featureValuesTest, correctLabels)))