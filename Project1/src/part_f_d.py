import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
import pandas as pd
from random import random, seed
import sys

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.linear_model import LinearRegression, Ridge, Lasso, lars_path

from project1_lib import FrankeFunction, make_franke_vecs, OLSRegression
from project1_lib import load_terrain_data, plot_terrain_data, MSE
from project1_lib import OLS_beta, plot_bias_variance_OLS, plot_bias_variance_ridge
from project1_lib import plot_ridge_heatmap_MSE, plot_ridge_heatmap_R2
from project1_lib import RidgeRegression, R2

"""
Plots the bias variance for different values of lambda and for different polynomial_degree
for Ridge regression on the terrain data.
Then plots the train MSE and R2.
"""

X, z = load_terrain_data(100)
X = scale(X)
z = scale(z)


#Plotting bias variance for different polynomial degrees.
plt.subplot(2,2,1)
plot_bias_variance_ridge(X,z, 10, 1)
plt.subplot(2,2,2)
plot_bias_variance_ridge(X,z, 10, 4)
plt.subplot(2,2,3)
plot_bias_variance_ridge(X,z, 10, 5)
plt.subplot(2,2,4)
plot_bias_variance_ridge(X,z, 10, 10)

plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.show()




max_degree = 10
num_lambda_values = 4
lambda_values = np.logspace(-2, 1, num_lambda_values)
print(lambda_values)
MSEs = []
R2s = []

MSEs_ridge = np.zeros((num_lambda_values, max_degree))
R2s_ridge = np.zeros((num_lambda_values, max_degree))

i = 0
#Finds the training MSE and R2.
for degree in range(1, max_degree+1):
    poly = PolynomialFeatures(degree)
    X2 = poly.fit_transform(X)
    names = poly.get_feature_names()
    lin_reg = OLSRegression()
    lin_reg.fit(X2, z)
    CI = (lin_reg.beta_conf_interval())
    beta = lin_reg.get_beta()
    var = lin_reg.beta_variance()
    y1 = CI[:,0]
    y2 = CI[:,1]
    j = 0
    for lam in lambda_values:
        ridge_reg = RidgeRegression(lam)
        ridge_reg.fit(X2, z)
        MSEs_ridge[j, i] = MSE(ridge_reg.predict(X2), z)
        R2s_ridge[j,i] =  R2(ridge_reg.predict(X2), z)
        j += 1
    j = 0
    MSEs.append(lin_reg.MSE_score(X2, z))
    R2s.append(lin_reg.R2_score(X2, z))
    i+=1


plt.subplot(121)

plt.plot(range(1,max_degree+1), MSEs, linestyle="--", marker = "o")
for j in range(0, len(lambda_values)):
    plt.plot(range(1,max_degree+1), MSEs_ridge[j,:], linestyle="--", marker = "o")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("Training MSE")
plt.legend(["OLS", "ridge, lam = 0.01", "ridge, lam = 0.1", "ridge, lam = 1", "ridge, lam = 10"])

plt.subplot(122)

plt.plot(range(1,max_degree+1), R2s, linestyle="--", marker = "o")
for j in range(0, len(lambda_values)):
    plt.plot(range(1,max_degree+1), R2s_ridge[j,:], linestyle="--", marker = "o")
plt.xlabel("Polynomial degree")
plt.ylabel("R2")
plt.title("Training R2")
plt.legend(["OLS", "ridge, lam = 0.01", "ridge, lam = 0.1", "ridge, lam = 1", "ridge, lam = 10"])

plt.subplots_adjust(wspace=0.3)
plt.show()

plt.show()
plot_ridge_heatmap_MSE(15, -6, 1, 10, X, z)
plot_ridge_heatmap_R2(15, -6, 1, 10, X, z)


def plot_ci_for_lambda(data, target, min_exp, max_exp, num_lambda_values, polynomial_degree, coeff):
    """
    Plots the CI for ridge regression for different values of lambda.
    Coeff: is which regression coefficients we want to look at the interval.
    """
    poly = PolynomialFeatures(polynomial_degree)
    X = poly.fit_transform(data)
    lambda_values = np.logspace(min_exp, max_exp, num_lambda_values)

    beta_values = []
    beta_values_upper_interval  =[]
    beta_values_lower_interval = []
    for lam in lambda_values:
        ridge_reg = RidgeRegression(lam)
        ridge_reg.fit(X, target)
        beta_values.append(ridge_reg.get_beta()[coeff])
        beta_values_lower_interval.append(ridge_reg.beta_conf_interval()[coeff,0])
        beta_values_upper_interval.append(ridge_reg.beta_conf_interval()[coeff,1])
    plt.plot(lambda_values, beta_values, linestyle="--", color="b")
    plt.plot(lambda_values, beta_values_lower_interval, linestyle="--", color="b")
    plt.plot(lambda_values,beta_values_upper_interval, linestyle="--", color="b")
    plt.fill_between(lambda_values, beta_values_lower_interval, beta_values_upper_interval)
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("coefficient")
    plt.grid(True)
    plt.legend(["Coefficient", "Upper limit", "Lower limit"])
    plt.show()

plot_ci_for_lambda(X, z, -10, 7, 10, 3, -1)
