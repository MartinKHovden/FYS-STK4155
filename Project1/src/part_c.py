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

from project1_lib import FrankeFunction, make_franke_vecs, OLSRegression, OLS_beta, plot_bias_variance_OLS, plot_bias_variance_ridge, RidgeRegression,MSE

"""
Does a k-fold cross validation and plots the bias, variance and the MSE. All found
by cross validation in the function. 
"""


noise = 0
X, z = make_franke_vecs(noise, 20)
X = scale(X)
z = scale(z)

plot_bias_variance_OLS(X, z, 20)
