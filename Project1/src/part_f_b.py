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

from project1_lib import FrankeFunction, make_franke_vecs, OLSRegression, OLS_beta, plot_bias_variance_OLS, plot_bias_variance_ridge, MSE, cv_ols
from project1_lib import load_terrain_data, plot_terrain_data

"""
Does a resample, fits the OLS model on the training data, and find the training and test error.
Also prints out the lowest train error found.
"""

X, z = load_terrain_data(100)
plot_terrain_data()
X = scale(X)
z = scale(z)

max_degree = 25

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2, shuffle=True)

MSE_train = np.zeros(max_degree+1)
MSE_test = np.zeros(max_degree+1)
i = 0

for degree in range(0,max_degree+1):
    poly = PolynomialFeatures(degree)
    design_test = poly.fit_transform(X_test)
    design_train = poly.fit_transform(X_train)
    reg = OLSRegression()
    reg.fit(design_train, z_train)

    MSE_train[i] = MSE(reg.predict(design_train), z_train)
    MSE_test[i] = MSE(reg.predict(design_test), z_test)

    i+=1

plt.plot(MSE_train, linestyle="--", marker="o")
plt.plot(MSE_test,  linestyle="--", marker="o")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.legend(["train", "test"])

plt.show()

print("Minimum MSE value: ", min(MSE_test))
