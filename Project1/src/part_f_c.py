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

"""
Does a k-fold cross validation and plots the bias, variance and the MSE. All found
by cross validation in the function.
"""


X, z = load_terrain_data(100)
X = scale(X)
z = scale(z)

plot_bias_variance_OLS(X, z, 20)
