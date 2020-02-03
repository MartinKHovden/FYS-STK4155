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

from project1_lib import FrankeFunction, make_franke_vecs, OLSRegression, OLS_beta, plot_bias_variance_OLS, plot_bias_variance_ridge
from project1_lib import load_terrain_data, plot_terrain_data

"""
Doing the same as was done for part_a.py, only for the terrain data.
"""

X, z = load_terrain_data(100)
plot_terrain_data()
X = scale(X)
z = scale(z)

max_degree = 5

MSEs = []
R2s = []

X = pd.DataFrame(X, columns = list("xy"))

#For loop iterates over the degrees, and computes confidence intervals and prints them out
for degree in range(1, max_degree+1):
    poly = PolynomialFeatures(degree)
    X2 = poly.fit_transform(X)
    lin_reg = OLSRegression()
    names = poly.get_feature_names()
    lin_reg.fit(X2, z)
    CI = (lin_reg.beta_conf_interval())
    beta = lin_reg.get_beta()
    var = lin_reg.beta_variance()
    print("Degree:" , degree)
    for i in range(0,len(names)):
        print("beta_%s:" %names[i],"%.3f" %beta[i], "CI:", CI[i,:])

    print("---------------------")
    MSEs.append(lin_reg.MSE_score(X2, z))
    R2s.append(lin_reg.R2_score(X2, z))


plt.subplot(121)
plt.plot(range(1,max_degree+1), MSEs, linestyle="--", marker = "o")
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.title("Training MSE")
plt.legend(["MSE"])

plt.subplot(122)
plt.plot(range(1,max_degree+1), R2s, linestyle="--", marker = "o")
plt.xlabel("Polynomial degree")
plt.ylabel("R2")
plt.title("Training R2")
plt.legend(["R2"])

plt.subplots_adjust(wspace=0.3)

plt.show()
