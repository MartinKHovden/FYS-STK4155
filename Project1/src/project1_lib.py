from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from imageio import imread
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
import pandas as pd
from random import random, seed
import sys
import scipy as sc
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, lars_path

import seaborn as sb

def FrankeFunction(x,y):
  term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
  term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
  term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
  term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
  return term1 + term2 + term3 + term4

def plot_Franke_function(x,y):
    z = FrankeFunction(x,y)

    surf = ax.plot_surface(x,y,z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

    ax.set_zlim(-0.1, 1.4)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def make_franke_vecs(noise, points):

    x = np.linspace(0,1,points)
    y = np.linspace(0,1,points)
    x,y = np.meshgrid(x,y)
    z = FrankeFunction(x,y)
    X = np.column_stack((np.ravel(x),np.ravel(y)))
    z = np.ravel(z)
    z = z + noise*np.random.randn(z.shape[0])
    return X, z

def plot_franke_function():
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    x = np.arange(0, 1, 0.005)
    y = np.arange(0, 1, 0.005)
    x,y = np.meshgrid(x,y)

    z = FrankeFunction(x,y)

    surf = ax.plot_surface(x,y,z, cmap = cm.coolwarm, linewidth = 0)#, antialiased = False)

    ax.set_zlim(-0.1, 1.4)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1,x2)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def MSE(z_pred, z):
    """
    Returns the MSE of the regression.
    """
    n = len(z_pred)
    return sum((z_pred - z)**2)*(1./n)

def R2(z_pred, z):
    """
    Returns the R2-score of the regression.
    """
    z_bar = np.mean(z_pred)
    return 1 - (sum((z - z_pred)**2)/sum((z - z_bar)**2))

def OLS_beta(design_matrix, target_values):
    """
    Returns the beta for the ordinary least squares. Uses the pinv numpy function.
    The pinv helps if the matrix becomes almost singular (or singular) and prevents
    the solution to blow up.
    """
    return np.linalg.pinv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(target_values)
#
# def SVD_OLS_beta(design_matrix, target_values):
#     """
#     Returns the neta for the ordinary least square. Uses the SVD composition
#     to improve stability.
#     """
#     u,s,v = sc.linalg.svd(design_matrix)
#     return v.T @ sc.linalg.pinv(sc.linalg.diagsvd(s, u.shape[0], v.shape[1])) @ u.T @ target_values

def ridge_regression_beta(design_matrix, target_values, lam):
    """
    Returns the beta for the ridge regression.
    """
    return np.linalg.inv(design_matrix.T.dot(design_matrix) + \
    lam*np.eye(len(design_matrix[1,:]))).dot(design_matrix.T).dot(target_values)

class OLSRegression:
    """
    A class for ordinary least squares.

    Attributes
    ----------
    fitted : int
        An int to check if model is fitted.

    Methods
    -------
    fit(design_matix, target)
        Fits the model and computes the beta.

    get_beta()
        Returns the beta if the model is fitted.

    predict(data)
        Returns the predicted values on the data set if the model is fitted.

    beta_variance()
        Returns the variance of the regression coefficients.

    R2_score(data, target)
        Returns the R2-score, given a new design matrix and the target values.
        Predicts new values on the new data, and compares to the target values.

    MSE_score(data, target)
        Returns the MSE-score, given a new design matrix and the target values.
        Predicts new values on the new data, and compares to the target values.
    """

    fitted = 0
    def __init__(self):
        fitted = 0

    def fit(self, design_matrix, target):
        """Fits the model with ridge regression given a design matrix and a target.

        Parameters
        ----------
        design_matrix: 2D numpy array
            Matrix containing samples and features.

        target: 1D numpy array
            Vector that contains the target values of the regression.
        """
        self.beta = OLS_beta(design_matrix, target)
        self.design_matrix = design_matrix
        self.target = target
        self.fitted = 1

    def get_beta(self):
        """Returns the beta of the ridge regression if model is fitted. Error if
        model not yet fitted

        Parameters
        ----------
        None

        Returns
        -------
        beta: 1D numpy array
            Contains the regression coefficients.
        """
        if self.fitted == 0:
            print("Model not fitted yet")
            sys.exit(1)
        else:
            return self.beta

    def predict(self, data):
        """Returns the predicted values of a data-set given as data.

        Parameters
        ----------
        data: 2D numpy array. Design matrix for new data to be predicted.

        Returns
        -------
        1D numpy vector. Contains the predicted values.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)

        else:
            return data.dot(self.beta)
    def beta_variance(self):
        """
        Returns the variance of the betas.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)
        else:
            X = self.design_matrix
            n = len(self.target)
            p = len(self.beta)

            y = self.target
            y_hat = self.predict(self.design_matrix)
            sigma_2 = (1./(n-p-1))*sum((y - y_hat)**2)
            return abs(sigma_2)*np.diag(np.linalg.inv(np.dot(X.T, X)))

    def R2_score(self, data, target):
        """
        Returns the R2 score given a data matrix and a vector of target values.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)
        else:
            return R2(self.predict(data), target)

    def MSE_score(self, data, target):
        """
        Returns the MSE score given a data matrix and a vector of target values.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)
        else:
            return MSE(self.predict(data), target)

    def beta_conf_interval(self):
        """
        Returns the confidence intervalls.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)
        else:
            n = len(self.beta)
            conf_int = np.stack((self.beta, self.beta), axis=-1)
            conf_int[:,0] -= 1.96*np.sqrt(self.beta_variance())
            conf_int[:,1] += 1.96*np.sqrt(self.beta_variance())

            return conf_int

class RidgeRegression:
    """
    A class for Ridge regression

    Attributes
    ----------
    fitted : int
        An int to check if model is fitted.

    Methods
    -------
    fit(design_matix, target)
        Fits the model and computes the beta.

    get_beta()
        Returns the beta if the model is fitted.

    predict(data)
        Returns the predicted values on the data set if the model is fitted.

    beta_variance()
        Returns the variance of the regression coefficients.

    R2_score(data, target)
        Returns the R2-score, given a new design matrix and the target values.
        Predicts new values on the new data, and compares to the target values.

    MSE_score(data, target)
        Returns the MSE-score, given a new design matrix and the target values.
        Predicts new values on the new data, and compares to the target values.
    """

    fitted = 0
    def __init__(self, lam):
        """
        Parameters
        ----------
        lam: int
            The penalty parameter lambda.
        """
        self.lam = lam

    def fit(self, design_matrix, target):
        """Fits the model with ridge regression given a design matrix and a target.

        Parameters
        ----------
        design_matrix: 2D numpy array
            Matrix containing samples and features.

        target: 1D numpy array
            Vector that contains the target values of the regression.
        """
        self.beta = ridge_regression_beta(design_matrix, target, self.lam)
        self.design_matrix = design_matrix
        self.target = target
        self.fitted = 1

    def get_beta(self):
        """Returns the beta of the ridge regression if model is fitted. Error if
        model not yet fitted

        Parameters
        ----------
        None

        Returns
        -------
        beta: 1D numpy array
            Contains the regression coefficients.
        """
        if self.fitted == 0:
            print("Model not fitted yet")
            sys.exit(1)
        else:
            return self.beta

    def predict(self, data):
        """Returns the predicted values of a data-set given as data.

        Parameters
        ----------
        data: 2D numpy array. Design matrix for new data to be predicted.

        Returns
        -------
        1D numpy vector. Contains the predicted values.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)

        else:
            return data.dot(self.beta)

    def beta_variance(self):
        """
        Returns the variance of the betas.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)
        else:
            X = self.design_matrix
            n = len(self.target)
            p = len(self.beta)
            y = self.target
            y_hat = self.predict(self.design_matrix)
            sigma_2 = (1./(n-p-1))*sum((y - y_hat)**2)
            return np.diag(  sigma_2*(np.linalg.inv(X.T @ X + self.lam*np.eye(len(X[1,:]))) @ X.T @ X @ (np.linalg.inv(X.T @ X + self.lam*np.eye(len(X[1,:])))).T )  )
            # return np.diag(sigma_2*(np.linalg.inv(np.dot(X.T, X))))

    def R2_score(self, data, target):
        """
        Returns the R2-score given a data_matrix and target values.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)
        else:
            return R2(self.predict(data), target)

    def MSE_score(self, data, target):
        """
        Comuptes the MSE given a data matrix and target values.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)
        else:
            return MSE(self.predict(data), target)

    def beta_conf_interval(self):
        """
        Returns the confidence interval for each beta.
        """
        if self.fitted== 0:
            print("Model not fitted yet")
            sys.exit(2)
        else:
            n = len(self.beta)
            conf_int = np.stack((self.beta, self.beta), axis=-1)
            conf_int[:,0] -= 1.96*np.sqrt(self.beta_variance())
            conf_int[:,1] += 1.96*np.sqrt(self.beta_variance())

            return conf_int


def k_fold_cv(num_folds, data, response, model, measure):
    """
    Returns the cv-error, calculated on k folds.

    Input:
    ----------------
    num_folds: integer value for number of folds in each step of the cv-computation.
    data: 2D numpy array. Conatins the covariates.
    response: 1D numpy array. Contains the response variable.
    model: Class LinearRegression or class RidgeRegression.
    measure: function. MSE or R2.


    Output:
    ----------------
    cv_error: 1D numpy array of CV-error for different degrees.
    """

    k_folds = KFold(n_splits = num_folds,shuffle=True)
    fold_score = np.zeros(num_folds)
    i = 0
    for train_index, test_index in k_folds.split(data):
        x_train = data[train_index]
        x_test = data[test_index]

        y_train = response[train_index]
        y_test = response[test_index]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        fold_score[i] = measure(y_pred, y_test)

        i+=1
    # return (1./num_folds)*np.mean(fold_score)
    return np.mean(fold_score)

def k_fold_cv_lasso(num_folds, data, response, model, measure):
    """
    Should only be used with Lasso as model due to the format of output for
    prediction.
    """
    k_folds = KFold(n_splits = num_folds,shuffle=True)
    fold_score = np.zeros(num_folds)
    i = 0
    for train_index, test_index in k_folds.split(data):
        x_train = data[train_index]
        x_test = data[test_index]

        y_train = response[train_index]
        y_test = response[test_index]

        model.fit(x_train, y_train)

        # y_pred = model.predict(x_test)

        fold_score[i] = measure(model.predict(x_test).reshape(-1,1), y_test.reshape(-1,1))

        i+=1

    return np.mean(fold_score)


def cv_ols(num_folds, data, response, max_degree):
    """"
    Function for computing the cv-errors for different degrees of polynomial in the
    OLS regression.

    Input:
    ----------------
    num_folds: integer value for number of folds in each step of the cv-computation.
    data: 2D numpy array. Conatins the covariates.
    response: 1D numpy array. Contains the response variable.
    max_lambda: interger. Highest value of lambda to compute CV for.

    Output:
    ----------------
    cv_error: 1D numpy array of CV-error for different degrees.
    """
    cv_error = np.zeros(max_degree)
    for i in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree = i)
        data_matrix = poly.fit_transform(data)
        reg = LinearRegression()

        cv_error[i-1] = k_fold_cv(num_folds, data_matrix, response, reg)
    return cv_error

def cv_ridge(num_folds, data, response, max_lambda):
    """"
    Function for computing the cv-errors for different values of lambda in the
    ridge regression.

    Input:
    ----------------
    num_folds: integer value for number of folds in each step of the cv-computation.
    data: 2D numpy array. Conatins the covariates.
    response: 1D numpy array. Contains the response variable.
    max_lambda: interger. Highest value of lambda to compute CV for.

    Output:
    ----------------
    cv_error: 1D numpy array of CV-error for different values of lambda.
    """
    cv_error = np.zeros(max_lambda)
    for i in range(1, max_lambda+1):
        ridge_temp = RidgeRegression(i)
        cv_error[i-1] = k_fold_cv(num_folds, data, response, ridge_temp)
    return cv_error

def cv_lasso(num_folds, data, response, max_lambda):
    """
    Function for computing the cv-errors for different values of lambda in the
    lasso regression.

    Input:
    ----------------
    num_folds: integer value for number of folds in each step of the cv-computation.
    data: 2D numpy array. Conatins the covariates.
    response: 1D numpy array. Contains the response variable.
    max_lambda: interger. Highest value of lambda to compute CV for.

    Output:
    ----------------
    cv_error: 1D numpy array of CV-error for different values of lambda.
    """
    cv_error = np.zeros(max_lambda)
    for i in range(1, max_lambda+1):
        ridge_temp = Lasso(i)
        cv_error[i-1] = k_fold_cv(num_folds, data, response, ridge_temp)
    return cv_error

def bias(response_values, predicted_values):
    """
    Returns the bias.
    Input:
    ------------------
    response_values: 1D numpy array. Contains the actual response values.
    predicted_values: 1D numpy array. Contains the predicted response values.

    Output:
    -------------------
    Numpy 1D array. Contains the bias.
    """
    return np.mean((response_values - np.mean(predicted_values))**2)

def plot_ridge_coefs(design_matrix, z):
    """
    Function for plotting the coefficients of the Lasso regression, for
    different values of lambda.

    Input:
    -----------------
    design_matrix: 2D numpy array containing the covariates.
    z: 1D numpy array conatining the response values.

    Output:
    ------------------
    Plot of the coeffiecients for different values of lambda
    """
    _,_,coefs = lars_path(design_matrix,z, method = "ridge", verbose=True)
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle="dashed")
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel("Coefficients")
    plt.title("Ridge coeffients")
    plt.axis("tight")
    plt.show()

def plot_lasso_coefs(design_matrix, z):
    """
    Function for plotting the coefficients of the Lasso regression, for
    different values of lambda.

    Input:
    -----------------
    design_matrix: 2D numpy array containing the covariates.
    z: 1D numpy array conatining the response values.

    Output:
    ------------------
    Plot of the coeffiecients for different values of lambda
    """
    _,_,coefs = lars_path(design_matrix,z, method = "lasso", verbose=True)
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle="dashed")
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel("Coefficients")
    plt.title("Lasso coeffients")
    plt.axis("tight")
    plt.show()

def plot_bias_variance_OLS(data, response, max_degree):
    """
    Function for plotting the bias variance trade off for OLS.
    """
    num_splits = 5

    bias_values = np.zeros(max_degree)
    variance = np.zeros(max_degree)
    total = np.zeros(max_degree)
    MSE_vals = np.zeros(max_degree)

    x_vals = np.linspace(0,max_degree, max_degree)

    X_train_1, X_test, z_train_1, z_test = train_test_split(data, response, test_size = 0.2)#,random_state = 42)

    l = len(z_test)

    for i in range(0, max_degree):
        k_folds = KFold(n_splits = num_splits, shuffle=True, random_state=1)

        fold_score = np.zeros(num_splits)

        poly = PolynomialFeatures(degree = i)
        data_matrix_train = poly.fit_transform(X_train_1)
        data_matrix_test = poly.fit_transform(X_test)
        reg = LinearRegression()

        predictions = np.zeros((l, num_splits))

        j = 0

        for train_index, test_index in k_folds.split(data_matrix_train):
            X_train_2 = data_matrix_train[train_index]
            # X_val = X_train_1[test_index]

            z_train_2 = z_train_1[train_index]
            # z_val = z_train[test_index]

            reg.fit(X_train_2, z_train_2)

            y_pred = reg.predict(data_matrix_test)
            # print(y_pred)

            predictions[:,j] = y_pred.T

            # fold_score[i] = MSE(y_pred, y_test)

            j+=1
        bias_values[i] = np.mean((z_test - np.mean(predictions, axis = 1))**2)
        variance[i] = np.mean(np.var(predictions, axis = 1))
        total[i] = bias_values[i] + variance[i]
        MSE_vals[i] = np.mean( np.mean((z_test[:,np.newaxis] - predictions)**2, axis=1, keepdims=True) )
    plt.plot(x_vals, variance, linestyle='--', marker="o")
    plt.plot(x_vals, bias_values, linestyle='--', marker="o")
    plt.plot(x_vals, MSE_vals, linestyle="--", marker = "o")
    plt.legend(["variance", "bias", "MSE"])
    plt.xlabel("Polynomial degree")
    plt.ylabel("")
    plt.show()
    print("min MSE:", min(MSE_vals))

def plot_bias_variance_ridge(data, response, num_lambda_vals, polynomial_degree):
    """
    Function for plotting the bias-variance trade-off using Ridge regression.
    Plots for different values of lamba between 1e-3 to 1e1.
    """
    num_splits = 5

    bias_values = np.zeros(num_lambda_vals)
    variance = np.zeros(num_lambda_vals)
    MSE_vals = np.zeros(num_lambda_vals)
    max_exp = 4
    min_exp = -2

    x_vals = np.logspace(min_exp,max_exp, num_lambda_vals)

    X_train_1, X_test, z_train_1, z_test = train_test_split(data, response, test_size = 0.2)#,random_state = 42)

    l = len(z_test)

    k = 0

    for i in np.logspace(min_exp,  max_exp, num_lambda_vals):#range(0, max_lambda):
        k_folds = KFold(n_splits = num_splits, shuffle=True, random_state=1)

        fold_score = np.zeros(num_splits)

        ridge_reg = RidgeRegression( lam= i)

        poly = PolynomialFeatures(degree = polynomial_degree, include_bias=False)
        data_matrix_train = poly.fit_transform(X_train_1)
        data_matrix_test = poly.fit_transform(X_test)

        predictions = np.zeros((l, num_splits))

        j = 0

        for train_index, test_index in k_folds.split(data_matrix_train):
            print(train_index)
            X_train_2 = data_matrix_train[train_index]
            # X_val = X_train_1[test_index]
            z_train_2 = z_train_1[train_index]



            # z_val = z_train[test_index]

            ridge_reg.fit(X_train_2, z_train_2)

            y_pred = ridge_reg.predict(data_matrix_test)
            # print(y_pred)

            predictions[:,j] = y_pred

            # fold_score[i] = MSE(y_pred, y_test)

            j+=1
        bias_values[k] = np.mean((z_test - np.mean(predictions, axis = 1))**2)
        variance[k] = np.mean(np.var(predictions, axis = 1))
        MSE_vals[k] = np.mean( np.mean((z_test[:,np.newaxis] - predictions)**2, axis=1, keepdims=True) )
        k+=1
    plt.plot(x_vals, variance, linestyle="--",marker="o")
    plt.plot(x_vals, bias_values, linestyle ="--", marker="o")
    plt.plot(x_vals, MSE_vals, linestyle="--", marker="o")
    plt.xscale("log")
    # plt.logplot(x_vals, variance, linestyle="--",marker="o")
    # plt.logplot(x_vals, bias_values, linestyle ="--", marker="o")
    # plt.logplot(x_vals, MSE_vals, linestyle="--", marker="o")
    plt.legend(["variance", "bias", "MSE"])
    plt.xlabel("lambda")
    plt.title("Bias-variance Ridge, degree = %.2f" %(polynomial_degree))
    print("min MSE:", min(MSE_vals))

    # plt.show()

def plot_bias_variance_lasso(data, response, num_lambda_vals, polynomial_degree):
    """
    Function for plotting the bias-variance trade-off using Ridge regression.
    Plots for different values of lamba between 1e-3 to 1e1.
    """
    num_splits = 5

    bias_values = np.zeros(num_lambda_vals)
    variance = np.zeros(num_lambda_vals)
    MSE_vals = np.zeros(num_lambda_vals)

    min_exp = -4
    max_exp = 2

    x_vals = np.logspace(min_exp,max_exp, num_lambda_vals)

    X_train_1, X_test, z_train_1, z_test = train_test_split(data, response, test_size = 0.2)#,random_state = 42)

    l = len(z_test)

    k = 0

    for i in np.logspace(min_exp, max_exp, num_lambda_vals):
        k_folds = KFold(n_splits = num_splits, shuffle=True)

        fold_score = np.zeros(num_splits)

        lasso_reg = Lasso(i, fit_intercept=False, tol = 0.01, max_iter=1e5)

        poly = PolynomialFeatures(degree = polynomial_degree, include_bias=False)
        data_matrix_train = poly.fit_transform(X_train_1)
        data_matrix_test = poly.fit_transform(X_test)

        predictions = np.zeros((l, num_splits))

        j = 0

        for train_index, test_index in k_folds.split(data_matrix_train):
            X_train_2 = data_matrix_train[train_index]
            # X_val = X_train_1[test_index]

            z_train_2 = z_train_1[train_index]
            # z_val = z_train[test_index]

            lasso_reg.fit(X_train_2, z_train_2)

            y_pred = lasso_reg.predict(data_matrix_test)
            # print(y_pred)

            predictions[:,j] = y_pred.T

            # fold_score[i] = MSE(y_pred, y_test)

            j+=1
        bias_values[k] = np.mean((z_test - np.mean(predictions, axis = 1))**2)
        variance[k] = np.mean(np.var(predictions, axis = 1))
        MSE_vals[k] = np.mean( np.mean((z_test[:,np.newaxis] - predictions)**2, axis=1, keepdims=True) )
        k+=1
    plt.plot(x_vals, variance, linestyle="--",marker="o")
    plt.plot(x_vals, bias_values, linestyle ="--", marker="o")
    plt.plot(x_vals, MSE_vals, linestyle="--", marker="o")
    plt.xscale("log")
    plt.legend(["variance", "bias", "MSE"])
    plt.xlabel("lambda")
    plt.title("Bias-variance lasso, degree = %.2f" %(polynomial_degree))
    print("min MSE:", min(MSE_vals))



def plot_ridge_heatmap_R2(max_degree, min_exp, max_exp, num_lambda_vals, data, target):
    """
    Plots a heatmap for the cross-validation R2 for ridge regression for varying
    lambda and polynomial degree.
    """
    R2s = np.zeros((max_degree, num_lambda_vals))
    j = 0
    for complexity in range(0,max_degree):
        for lam in np.logspace(min_exp, max_exp,num= num_lambda_vals):
            poly = PolynomialFeatures(complexity, include_bias=False)
            design_matrix = poly.fit_transform(data)

            ridge_reg = RidgeRegression(lam)

            R2s[complexity, j] = k_fold_cv(5, design_matrix, target, ridge_reg, R2)
            j+=1
        j = 0
    R2s = pd.DataFrame(R2s, columns=np.round(np.logspace(min_exp, max_exp, num_lambda_vals), 6))
    heatmat = sb.heatmap(R2s, linewidths=0.5, annot=True)
    plt.title("R2 for ridge regression")
    plt.xlabel("lambda")
    plt.ylabel("polynomial degree")
    plt.show()

def plot_ridge_heatmap_MSE(max_degree, min_exp, max_exp, num_lambda_vals, data, target):
    """
    Plots a heatmap for the cross-validation MSE for ridge regression for varying
    lambda and polynomial degree.
    """
    MSEs = np.zeros((max_degree, num_lambda_vals))
    j = 0
    for complexity in range(0,max_degree):
        for lam in np.logspace(min_exp, max_exp,num= num_lambda_vals):
            poly = PolynomialFeatures(complexity, include_bias=False)
            design_matrix = poly.fit_transform(data)

            ridge_reg = RidgeRegression(lam)

            MSEs[complexity, j] = k_fold_cv(5, design_matrix, target, ridge_reg, MSE)

            j+=1
        j = 0
    MSEs = pd.DataFrame(MSEs, columns=np.round(np.logspace(min_exp, max_exp, num_lambda_vals),6))
    heatmat = sb.heatmap(MSEs, linewidths=0.5, annot=True)
    plt.title("MSE for ridge regression")
    plt.xlabel("lambda")
    plt.ylabel("polynomial degree")
    plt.show()
    print("min MSE:", min(MSEs))

def plot_lasso_heatmap_MSE(max_degree, min_exp, max_exp, num_lambda_vals, data, target):
    """
    Plots a heatmap for the cross-validation MSE for lasso regression for varying
    lambda and polynomial degree.
    """
    MSEs = np.zeros((max_degree, num_lambda_vals))
    j = 0
    for complexity in range(1,max_degree+1):
        for lam in np.logspace(min_exp, max_exp,num= num_lambda_vals):
            poly = PolynomialFeatures(complexity, include_bias=False)
            design_matrix = poly.fit_transform(data)

            lasso_reg = Lasso(lam, fit_intercept=False, tol = 0.01, max_iter = 1e5)
            # ridge_reg.fit(design_matrix, target)
            MSEs[complexity-1, j] = k_fold_cv_lasso(5, design_matrix, target, lasso_reg, MSE)
            # MSEs[complexity, j] = MSE(ridge_reg.predict(design_matrix), target)
            j+=1
        j = 0
    # MSEs = MSEs[1:,:]
    MSEs = pd.DataFrame(MSEs, columns=np.round(np.logspace(min_exp, max_exp, num_lambda_vals),6), index=range(1,max_degree+1))
    heatmat = sb.heatmap(MSEs, linewidths=0.5, annot=True)
    print(MSEs)
    plt.title("MSE for lasso regression")
    plt.xlabel("lambda")
    plt.ylabel("polynomial degree")
    plt.show()
    print("min MSE:", min(MSEs))


def plot_lasso_heatmap_R2(max_degree, min_exp, max_exp, num_lambda_vals, data, target):
    """
    Plots a heatmap for the cross-validation R2 for lasso regression for varying
    lambda and polynomial degree.
    """
    R2s = np.zeros((max_degree, num_lambda_vals))
    j = 0
    for complexity in range(1,max_degree+1):
        for lam in np.logspace(min_exp, max_exp,num= num_lambda_vals):#range(0,1, 10):
            poly = PolynomialFeatures(complexity, include_bias=False)
            design_matrix = poly.fit_transform(data)

            lasso_reg = Lasso(lam, fit_intercept=False, tol = 0.01, max_iter = 1e5)
            # ridge_reg.fit(design_matrix, target)
            # R2s[complexity, j] = R2(ridge_reg.predict(design_matrix), target)
            R2s[complexity-1, j] = k_fold_cv_lasso(5, design_matrix, target, lasso_reg, R2)
            j+=1
        j = 0
    # R2s = R2s[1:,:]
    R2s = pd.DataFrame(R2s, columns=np.round(np.logspace(min_exp, max_exp, num_lambda_vals),6), index = range(1,max_degree+1))
    heatmat = sb.heatmap(R2s, linewidths=0.5, annot=True)
    plt.title("R2 for lasso regression")
    plt.xlabel("lambda")
    plt.ylabel("polynomial degree")
    plt.show()

def load_terrain_data(num_pixels):
    """
    Loads the terrain data.
    Loads a num_pixels x num_pixels grid.
    """
    n = num_pixels
    terrain1 = imread("n59_e010_1arc_v3.tif")
    terrain1 = np.array(terrain1)[600:600+n, 300:300+n]
    x = np.linspace(0,terrain1.shape[0], terrain1.shape[0])
    y = np.linspace(0, terrain1.shape[1], terrain1.shape[1])
    x,y = np.meshgrid(x,y)
    X = np.column_stack((np.ravel(x),np.ravel(y)))
    z = np.ravel(terrain1)
    # X = X[:n,:]
    # z = z[:n]
    return X, z

def plot_terrain_data():
    """
    Function for plotting the terrain data.
    Plots a 100x100 grid.
    """
    terrain1 = imread("n59_e010_1arc_v3.tif")
    terrain1 = np.array(terrain1)[600:700,300:400]
    plt.figure()
    plt.title("Terrain over Norway")
    sns.heatmap(terrain1)
    # plt.imshow(terrain1, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
