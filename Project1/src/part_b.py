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

from project1_lib import FrankeFunction, make_franke_vecs, OLSRegression, OLS_beta, plot_bias_variance_OLS, plot_bias_variance_ridge, MSE,cv_ols, RidgeRegression

"""
Does a resampling and splits the data into a training part and a test part. Then
computes the training error and the test error and plots them. 
"""
noise = 0 #sigma^2

max_degree = 15

X, z = make_franke_vecs(noise, 20)
X = scale(X)
z = scale(z)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2, shuffle=True)

MSE_train = np.zeros(max_degree+1)
MSE_test = np.zeros(max_degree+1)
i = 0

for degree in range(0,max_degree+1):
    poly = PolynomialFeatures(degree)
    design_test = poly.fit_transform(X_test)
    design_train = poly.fit_transform(X_train)
    #
    reg = OLSRegression()
    reg.fit(design_train, z_train)
    # ridge = RidgeRegression(1e-7)
    # ridge.fit(design_train, z_train)

    MSE_train[i] = MSE(reg.predict(design_train), z_train)
    MSE_test[i] = MSE(reg.predict(design_test), z_test)

    i+=1

plt.plot(MSE_train, linestyle="--", marker="o")
plt.plot(MSE_test,  linestyle="--", marker="o")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.legend(["train MSE", "test MSE"])

plt.show()

print(min(MSE_test))
