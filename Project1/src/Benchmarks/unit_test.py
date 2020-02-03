import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

import numpy as np
import pandas as pd
from random import random, seed
import sys

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.linear_model import LinearRegression, Ridge, Lasso, lars_path

from project1_lib import FrankeFunction, make_franke_vecs, OLSRegression, RidgeRegression
from project1_lib import load_terrain_data, plot_terrain_data, MSE, LinearRegression
from project1_lib import OLS_beta, plot_bias_variance_OLS, plot_bias_variance_ridge

def test_ols_precision():
    """
    Testes if my code gives the same as sklearn
    """
    X, z = make_franke_vecs(0, 20)
    lin_reg_test = OLSRegression()
    lin_reg_test.fit(X, z)

    lin_reg_sklearn = LinearRegression(fit_intercept=False)
    lin_reg_sklearn.fit(X,z)

    lin_reg_test_beta = lin_reg_test.get_beta()
    lin_reg_sklearn_beta = lin_reg_sklearn.coef_

    print(lin_reg_test_beta, lin_reg_test_beta)

    if sum(lin_reg_test_beta - lin_reg_sklearn_beta) < 0.1:
        return_value = 1
    else:
        return_value = 0

    poly = PolynomialFeatures(4)
    X = poly.fit_transform(X)

    lin_reg_test.fit(X, z)
    lin_reg_sklearn.fit(X,z)

    lin_reg_test_beta = lin_reg_test.get_beta()
    lin_reg_sklearn_beta = lin_reg_sklearn.coef_

    # print(lin_reg_test_beta, lin_reg_test_beta)

    if sum(lin_reg_test_beta - lin_reg_sklearn_beta) < 0.1:
        return_value *= 1
    else:
        return_value *= 0

    return return_value

def test_ridge_precision():
    """
    Testes if my code gives the same as sklearn
    """
    X, z = make_franke_vecs(0, 20)
    ridge_reg_test = RidgeRegression(1)
    ridge_reg_test.fit(X, z)

    ridge_reg_sklearn = Ridge(1,fit_intercept=False)
    ridge_reg_sklearn.fit(X,z)

    ridge_reg_test_beta = ridge_reg_test.get_beta()
    ridge_reg_sklearn_beta = ridge_reg_sklearn.coef_

    print(ridge_reg_test_beta, ridge_reg_test_beta)

    if sum(ridge_reg_test_beta - ridge_reg_sklearn_beta) < 0.1:
        return_value = 1
    else:
        return_value = 0

    poly = PolynomialFeatures(4)
    X = poly.fit_transform(X)

    ridge_reg_test.fit(X, z)
    ridge_reg_sklearn.fit(X,z)

    ridge_reg_test_beta = ridge_reg_test.get_beta()
    ridge_reg_sklearn_beta = ridge_reg_sklearn.coef_

    # print(ridge_reg_test_beta, ridge_reg_test_beta)

    if sum(ridge_reg_test_beta - ridge_reg_sklearn_beta) < 0.1:
        return_value *= 1
    else:
        return_value *= 0

    return return_value

if test_ols_precision()*test_ridge_precision() == 1:
    print("Everything seems to be fine with the code!")
else:
    print("Something is wrong with the code")
    sys.exit()

def time_ols():
    """
    Testing the time used to fit the model as a function of number of points.
    """
    n_values = np.linspace(10, 2000, 10)
    test_times = []
    sklearn_times = []
    for n in n_values:
        X, z = make_franke_vecs(0, n)
        lin_reg_test = OLSRegression()

        test_start = time.time()
        lin_reg_test.fit(X, z)
        test_end = time.time()

        lin_reg_sklearn = LinearRegression(fit_intercept=False)
        sklearn_start = time.time()
        lin_reg_sklearn.fit(X,z)
        sklearn_end = time.time()

        test_time = test_end - test_start
        test_times.append(test_time)
        sklearn_time = sklearn_end - sklearn_start
        sklearn_times.append(sklearn_time)
        print("Own code time:", test_time, "Sklearn time:", sklearn_time )
    plt.plot(n_values, test_times, linestyle="--", marker = "o")
    plt.plot(n_values, sklearn_times, linestyle="--", marker = "o")
    plt.xlabel("n")
    plt.ylabel("time")
    plt.title("Comparison between own code and scikit-learn for OLS")
    plt.legend(["My code", "Sklearn"])
    plt.show()

time_ols()

def time_ridge():
    """
    Testing the time used to fit the model as a function of number of points.
    """
    n_values = np.linspace(10, 2000, 10)
    test_times = []
    sklearn_times = []
    for n in n_values:
        X, z = make_franke_vecs(0, n)
        ridge_reg_test = RidgeRegression(1)

        test_start = time.time()
        ridge_reg_test.fit(X, z)
        test_end = time.time()

        ridge_reg_sklearn = Ridge(1, fit_intercept=False)
        sklearn_start = time.time()
        ridge_reg_sklearn.fit(X,z)
        sklearn_end = time.time()

        test_time = test_end - test_start
        test_times.append(test_time)
        sklearn_time = sklearn_end - sklearn_start
        sklearn_times.append(sklearn_time)
        print("Own code time:", test_time, "Sklearn time:", sklearn_time )
    plt.plot(n_values, test_times, linestyle="--", marker = "o")
    plt.plot(n_values, sklearn_times, linestyle="--", marker = "o")
    plt.xlabel("n")
    plt.ylabel("time")
    plt.title("Comparison between own code and scikit-learn for Ridge")
    plt.legend(["My code", "Sklearn"])
    plt.show()

time_ridge()
