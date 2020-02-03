import numpy as np
import pandas as pd
import time

from numba import jit

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures,StandardScaler, OneHotEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.compose import ColumnTransformer

from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_cumulative_gain

def load_credit_card_data():
    """Reads the data and returns the design matrix and the target values.

    Returns
    -------
    design_matrix: 2D numpy array
        Features and samples collected in a matrix.
        Dimension is (n_samples, n_features)

    targets: 1D numpy array
        Targets for each sample.
        Dimension is (n_samples)
    """

    nanDict = {}
    data = pd.read_excel("default_of_credit_card_clients.xls", header=1, skiprows=0, index_col=0, na_values=nanDict)
    design_matrix = data.drop(["default payment next month"], axis=1)

    targets = data["default payment next month"]
    targets = np.expand_dims(targets, axis = 1)
    targets = np.reshape(targets, newshape=(-1,1))

    data = design_matrix


    #Fixing the unlabeled data.
    id_5 = data["EDUCATION"] == 5
    id_6 = data["EDUCATION"] == 6
    id_0 = data["EDUCATION"] == 0
    temp = id_5 | id_6 | id_0
    data.loc[temp, "EDUCATION"] = 4

    id_0 = data["MARRIAGE"] == 0
    data.loc[id_0, "MARRIAGE"] = 3

    pay_0_index = (data["PAY_0"] == -2) | (data["PAY_0"] == 0)
    pay_2_index = (data["PAY_2"] == -2) | (data["PAY_2"] == 0)
    pay_3_index = (data["PAY_3"] == -2) | (data["PAY_3"] == 0)
    pay_4_index = (data["PAY_4"] == -2) | (data["PAY_4"] == 0)
    pay_5_index = (data["PAY_5"] == -2) | (data["PAY_5"] == 0)
    pay_6_index = (data["PAY_6"] == -2) | (data["PAY_6"] == 0)
    data.loc[pay_0_index, "PAY_0"] = -1
    data.loc[pay_2_index, "PAY_2"] = -1
    data.loc[pay_3_index, "PAY_3"] = -1
    data.loc[pay_4_index, "PAY_4"] = -1
    data.loc[pay_5_index, "PAY_5"] = -1
    data.loc[pay_6_index, "PAY_6"] = -1


    onehotencoder = OneHotEncoder(categories="auto")

    data = ColumnTransformer(
    [("", onehotencoder, [3]),],
    remainder="passthrough"
    ).fit_transform(data)


    return data, targets

def load_breast_cancer_data():
    """Reads the data from the breast cancer data sets, and returns the data matrix
    and the target values.
    """

    data = load_breast_cancer()

    # sc = StandardScaler()
    # X = sc.fit_transform(data.data)
    X = data.data

    y = data.target
    y = np.reshape(y, newshape=(-1,1))

    return X, y

def sigmoid(beta, design_matrix):
    """Returns the sigmoid function

    Parameters
    ----------
    design_matrix : Numpy vector or int.

    Returns
    --------
    Numpy vector or int.
    """

    return 1./(1. + np.exp(-np.dot(design_matrix, beta)))

def sigmoid_test():
    """Different test cases for sigmoid. If all tests pass, returns true.
    """

    test1 = (sigmoid(1,0) == 0.5)
    test2 = (sigmoid(0,1) == 0.5)
    test3 = (sigmoid(0,0) == 0.5)
    test4 = (sigmoid(1,1) == 1/(1. + np.exp(-1)))

    if test1 and test2 and test3 and test4:
        return True
    else:
        return False

class LogReg:
    """
    Logistic regression class.
    """
    def __init__(self , iterations, alpha = 0.0001, intercept = True):
        self.iterations = iterations
        self.alpha = alpha
        self.intercept = intercept

    def fit(self, X, y, verbose = False, plot_cost_over_epoch = False):
        """
        Fits the logistic regression for the design matrix X and the target values y.

        Parameters
        -----------
        X: numpy array
            Data matrix
        y: numpy array
            Target values
        Iterations: Int
            Number of iterations in the gradient descent
        alpha: float
            Learning rate in the gradient descent method.
        intercept: boolean
            Wheter or not to fit with an intercept
        """

        #Adds an intercept to the design matrix if True.
        if self.intercept:
            poly = PolynomialFeatures(degree=1)
            X = poly.fit_transform(X)

        #Initialization of beta values
        self.beta_values = np.zeros((X.shape[1],1))

        cost_over_epochs = np.zeros(self.iterations)

        #Iterations of the gradient descent.
        for _ in range(self.iterations):
            z = np.dot(X, self.beta_values)
            p = self.sigmoid(z)
            cost_gradient = -np.dot(X.T, (y - p))
            cost = self.cost_function(self.sigmoid(np.dot(X, self.beta_values)), y)
            cost_over_epochs[_]=cost
            self.beta_values = self.beta_values - self.alpha*cost_gradient
            if (_%10 == 0) and verbose:
                print("cross-entropy: ", cost)
        if plot_cost_over_epoch:
            plt.plot(cost_over_epochs, linestyle="--")
            plt.title("Cost over epoch")
            plt.xlabel("epoch")
            plt.ylabel("Cross entropy")
            plt.show()
        return cost_over_epochs


    def probabilities(self, X):
        """
        Predicts the probabilitys of class 1 for the input matrix.

        Parameters
        ----------
        X: 2d numpy array
            Data matrix. Rows: samples, Columns: features.

        Returns
        --------
        numpy vector
            Probability for class 1 for each sample.
        """
        #Adds intercept if True
        if self.intercept:
            poly = PolynomialFeatures(degree=1)
            X = poly.fit_transform(X)

        p = self.sigmoid(np.dot(X, self.beta_values))  #Probability of class 1.
        return p

    def predict(self, X, threshold = 0.5):
        """
        Predicts the actual class using the porbabilities for belonging to
        class 1.

        Parameters
        -------
        X: numpy 2D array
            Data matrix. Rows: samples, Columns: features.

        Returns
        --------
        numpy 2D array
            Classes for each sample.
        """
        p = self.probabilities(X)            #Probability of class 1.
        classes = np.where(p >=threshold, 1, 0)    #If prob of class 1 > 0.5, class = 1. Else: class = 0.
        return classes

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def cost_function(self, probability, y):
        cost = -np.mean(y*np.log(probability) + (1-y)*np.log(1 - probability))
        return cost

def cv_classification_scores_logreg(num_splits, data, targets, model):
    """ Calculates the k-fold cross validation scores for AUC and accuracy.

    Parameters
    ----------
    num_splits: int
        number of folds in the cross validation.

    data: 2D numpy array
        Design matrix.

    targets: 2D numpy array
        Targets for each sample.

    model: LogReg
        Initialized logistic regression from the class LogReg.

    Returns
    -------
    cv_auc: float
        cross validation auc score.

    cv_accuracy: float
        cross validation accuracy score.
    """

    k_fold = KFold(n_splits = num_splits, shuffle=True)

    cv_auc_scores = np.zeros(num_splits)
    cv_accuracy_scores = np.zeros(num_splits)
    cv_area_scores = np.zeros(num_splits)

    i=0
    for train_index, test_index in k_fold.split(data):
        sc2 = StandardScaler()

        x_train = data[train_index]
        x_test = data[test_index]

        x_train = sc2.fit_transform(x_train)
        x_test = sc2.transform(x_test)

        y_train = targets[train_index]
        y_test = targets[test_index]

        model.fit(x_train, y_train)

        proba = model.probabilities(x_test)

        temp = np.concatenate((1-proba, proba), axis=1)

        ax = plot_cumulative_gain(y_test, temp)
        plt.close()
        lines = ax.lines
        cm =lines[1]
        model_ydata = cm.get_ydata()
        model_xdata = cm.get_xdata()

        area_model_curve = np.trapz(model_ydata, model_xdata)

        area_optimal_curve = 0.788 + 0.5*0.2212

        area_baseline_curve = 0.5

        area_ratio = (area_model_curve-area_baseline_curve)/(area_optimal_curve - area_baseline_curve)

        cv_auc_scores[i] = roc_auc_score(y_test, proba)
        cv_accuracy_scores[i] = accuracy_score(y_test, model.predict(x_test))
        cv_area_scores[i] = area_ratio
        i+=1
    cv_auc = np.mean(cv_auc_scores)
    cv_accuracy = np.mean(cv_accuracy_scores)
    cv_area_ratio = np.mean(cv_area_scores)

    print(cv_auc)
    print(cv_accuracy)
    print(cv_area_ratio)

    return cv_area_ratio, cv_auc, cv_accuracy


if __name__ == "__main__":
    print(" ")
