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

from project1_lib import FrankeFunction, make_franke_vecs, OLSRegression, OLS_beta, plot_bias_variance_OLS, plot_bias_variance_ridge
from project1_lib import plot_franke_function

"""
This file computed the training MSE and the training R2 for the franke function.
It also calculates the variance of each coefficient for polynomial regression
up to degree 5. It also find the confidence intervals of each coefficient.
"""

max_degree = 30

MSEs = []
R2s = []

noise = 0

X, z = make_franke_vecs(noise, 20)
X = pd.DataFrame(X, columns = list("xy"))
X =scale(X)
z = scale(z)

#Finding the trianing MSE and R2 for each polynomial degree.
#ALso calculates the confidence intervals and beta.
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
    # print(names)
    # print(beta)
    # print(CI)
    print("Degree:" , degree)
    for i in range(0,len(names)):
        print("beta_%s:" %names[i],"%.3f" %beta[i], "CI:", CI[i,:])

    print("---------------------")
    if degree > 1 and degree <=5:
        plt.subplot(4,1,degree-1)
        plt.errorbar(names,y = beta, yerr = 1.96*np.sqrt(var), fmt="none" , capsize=3)
        plt.grid(True)
    MSEs.append(lin_reg.MSE_score(X2, z))
    R2s.append(lin_reg.R2_score(X2, z))

plt.suptitle("Beta confidence intervalls for polynomial degree one to five")
plt.show()

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

print("Minimium MSE: ",min(MSEs))
print("Minimum R2:" ,max(R2s))
