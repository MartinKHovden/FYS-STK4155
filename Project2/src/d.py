from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from project1_lib import make_franke_vecs

from nn import NN, Layer, cv_regression_scores_nn, plot_bias_variance_nn_reg

#Reading the data
data, target = make_franke_vecs(0.3, 20)
target = np.expand_dims(target, axis=1)

plot_bias_variance_nn_reg(data, target, 6, 0.001)
plt.show()

#Setting up each layer
l1 = Layer(data.shape[1], 20,activation_function= "sigmoid")
l2 = Layer(20, 10, activation_function = "sigmoid")
l3 = Layer(10,5,activation_function = "sigmoid")
l4 = Layer(5,1,activation_function = None)

#Setting up the nerual network and adds each layer.
nnet = NN(cost_function="mse")
nnet.add_layer(l1)
nnet.add_layer(l2)
nnet.add_layer(l3)
nnet.add_layer(l4)


#FINDING THE BEST NUMBER OF ITERATIONS.
train_error = []
test_error = []

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size = 0.2)

epoch_values = [1, 100, 200, 300,  400,  500, 600, 700, 800, 900, 1000]

#Loops over all epoch values.
for epochs in epoch_values:
    nnet.train(train_data, train_target, epochs, 50, 0.01,0)

    train_pred = nnet.predict(train_data)
    test_pred = nnet.predict(test_data)

    train_error.append(mean_squared_error(train_pred, train_target))
    test_error.append(mean_squared_error(test_pred, test_target))

plt.plot(epoch_values, train_error, linestyle="--", marker = "o")
plt.plot(epoch_values, test_error, linestyle="--", marker = "o")
plt.legend(["train error", "test_error"])
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.show()

#GRID SEARCH FOR FINDING THE OPTIMAL PARAMETERS
n_lr_values = 5      #Number of learning rate values in grid search.
lr_values = np.logspace(-6,-2, n_lr_values)   #Vector of learning rate values.

n_reg_lam_values = 6      #Number of regularization rate parameteres in grid search.
reg_lam_values = np.logspace(-6,-1,n_reg_lam_values)   #Vector of reg param values.

mse_scores = np.zeros(shape=(n_lr_values, n_reg_lam_values))

#Searching thourgh the grid and updates the MSE matrix.
for i, learning_rate in enumerate(lr_values):
    for j,reg_lam in enumerate(reg_lam_values):
        cv_mse = cv_regression_scores_nn(5, data, target, nnet, 500, 50, learning_rate, reg_lam)
        mse_scores[i,j] = cv_mse

#Converts to pandas dataframe to make plotting easier.
mse_scores = pd.DataFrame(mse_scores, columns=np.round(reg_lam_values,6), index =np.round(lr_values,6) )

ax = sns.heatmap(mse_scores, annot=True, fmt=".4f", linewidths=0.5, cbar_kws={"label": "mse score" })
plt.ylim(top = 0, bottom=n_lr_values)
plt.xlabel("regularization parameter")
plt.ylabel("learning rate")

plt.show()



# nnet.train(data, target, 1000, 500, 0.001)
# pred = nnet.predict(data)
# print(pred.shape)
# print("MSE: ", np.mean((nnet.predict(data) - target)**2))
