from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
import pandas as pd
from random import random, seed
import sys

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.linear_model import LinearRegression, Ridge, Lasso, lars_path

from project1_lib import FrankeFunction, make_franke_vecs, OLSRegression, OLS_beta, plot_lasso_coefs
from project1_lib import plot_bias_variance_OLS, plot_bias_variance_ridge, RidgeRegression,MSE
from project1_lib import plot_bias_variance_lasso, R2, plot_lasso_heatmap_MSE,plot_lasso_heatmap_R2

"""
Plots the bias variance for different values of lambda and for different polynomial_degree
for Lasso regression.
Then plots the train MSE and R2.
Last plots heatmaps with cv-error estimates. 
"""

noise = 0.3
X_, z_ = make_franke_vecs(noise, 20)
X_ = scale(X_)
z_ = scale(z_)

#Plotting a 2x2 plot with lassos  bias variance for different polynomial degrees.
plt.subplot(2,2,1)
plot_bias_variance_lasso(X_,z_, 10, 1)
plt.subplot(2,2,2)
plot_bias_variance_lasso(X_,z_, 10, 4)
plt.subplot(2,2,3)
plot_bias_variance_lasso(X_,z_, 10, 5)
plt.subplot(2,2,4)
plot_bias_variance_lasso(X_,z_, 10, 10)

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.show()

max_degree = 10

num_lambda_values = 4
lambda_values = np.logspace(-2, 1, num_lambda_values) #Lambda values I want to plot for

#Two vectors for the training MSE and R2 for different polynomial degrees.
MSEs = []
R2s = []

#The two matrices for trianing MSE and R2 for different values of lambda and degree for lasso.
MSEs_lasso = np.zeros((num_lambda_values, max_degree))
R2s_lasso = np.zeros((num_lambda_values, max_degree))

X, z = make_franke_vecs(noise)
X = pd.DataFrame(X, columns = list("xy"))
X = scale(X)
z = scale(z)

i = 0
#Iterating over the degrees and lambdas, and updates the matrices.
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
        lasso_reg = Lasso(lam)
        lasso_reg.fit(X2, z)
        MSEs_lasso[j, i] = MSE(lasso_reg.predict(X2).reshape(-1,1), z.reshape(-1,1))
        R2s_lasso[j,i] =  R2(lasso_reg.predict(X2).reshape(-1,1), z.reshape(-1,1))
        j += 1
    j = 0
    MSEs.append(lin_reg.MSE_score(X2, z))
    R2s.append(lin_reg.R2_score(X2, z))
    i+=1


plt.subplot(121)

plt.plot(range(1,max_degree+1), MSEs, linestyle="--", marker = "o")
for j in range(0, len(lambda_values)):
    plt.plot(range(1,max_degree+1), MSEs_lasso[j,:], linestyle="--", marker = "o")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("Training MSE")
plt.legend(["OLS", "lasso, lam = 0.01", "lasso, lam = 0.1", "lasso, lam = 1", "lasso, lam = 10"])

plt.subplot(122)

plt.plot(range(1,max_degree+1), R2s, linestyle="--", marker = "o")
for j in range(0, len(lambda_values)):
    plt.plot(range(1,max_degree+1), R2s_lasso[j,:], linestyle="--", marker = "o")
plt.xlabel("Polynomial degree")
plt.ylabel("R2")
plt.title("Training R2")
plt.legend(["OLS", "lasso, lam = 0.01", "lasso, lam = 0.1", "lasso, lam = 1", "lasso, lam = 10"])

plt.subplots_adjust(wspace=0.3)
plt.show()

#Plotting the heatmap for lasso. Calculated with cross validation
plot_lasso_heatmap_R2(10, -3, 2, 10, X, z_)
plot_lasso_heatmap_MSE(10,-3, 0, 10, X, z_)
