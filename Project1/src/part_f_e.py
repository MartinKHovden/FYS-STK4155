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
from project1_lib import load_terrain_data, plot_terrain_data, MSE, R2, plot_lasso_coefs
from project1_lib import OLS_beta, plot_bias_variance_OLS, plot_bias_variance_ridge
from project1_lib import plot_bias_variance_lasso, plot_lasso_heatmap_R2, plot_lasso_heatmap_MSE

"""
Plots the bias variance for different values of lambda and for different polynomial_degree
for Lasso regression.
Then plots the train MSE and R2.
At last, plots heatmaps with cv-error estimates. 
"""

X, z = load_terrain_data(100)
X = scale(X)
z = scale(z)

plt.subplot(2,2,1)
plot_bias_variance_lasso(X,z, 10, 1)
plt.subplot(2,2,2)
plot_bias_variance_lasso(X,z, 10, 4)
plt.subplot(2,2,3)
plot_bias_variance_lasso(X,z, 10, 5)
plt.subplot(2,2,4)
plot_bias_variance_lasso(X,z, 10, 10)

plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.show()


max_degree = 10
num_lambda_values = 4
lambda_values = np.logspace(-2, 1, num_lambda_values)
print(lambda_values)
MSEs = []
R2s = []

MSEs_lasso = np.zeros((num_lambda_values, max_degree))
R2s_lasso = np.zeros((num_lambda_values, max_degree))

X = pd.DataFrame(X, columns = list("xy"))
X = scale(X)
z = scale(z)

i = 0
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
    # print(names)
    # plt.errorbar(range(len(y1)),y = beta, yerr = 1.96*np.sqrt(var), fmt="o" )
    # plt.show()
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
# plt.title("Training MSE for different polynomial degrees on Franke function")
plt.legend(["OLS", "lasso, lam = 0.01", "lasso, lam = 0.1", "lasso, lam = 1", "lasso, lam = 10"])

plt.subplot(122)

plt.plot(range(1,max_degree+1), R2s, linestyle="--", marker = "o")
for j in range(0, len(lambda_values)):
    plt.plot(range(1,max_degree+1), R2s_lasso[j,:], linestyle="--", marker = "o")
plt.xlabel("Polynomial degree")
plt.ylabel("R2")
plt.title("Training R2")
# plt.title("Training R2 for different polynomial degrees on Franke function")
plt.legend(["OLS", "lasso, lam = 0.01", "lasso, lam = 0.1", "lasso, lam = 1", "lasso, lam = 10"])

plt.subplots_adjust(wspace=0.3)
plt.show()

plot_lasso_heatmap_MSE(10, -7, 1, 10, X, z)
plot_lasso_heatmap_R2(10, -7, 1, 10, X, z)

X2 = np.copy(X)
poly2 = PolynomialFeatures(5)
X2 = poly2.fit_transform(X2)

plot_lasso_coefs(X2, z)
