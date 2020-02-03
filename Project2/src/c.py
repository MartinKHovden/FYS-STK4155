from nn import NN, Layer, cv_classification_scores_nn
from library import load_breast_cancer_data, load_credit_card_data

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_cumulative_gain

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

#Loads the data
data, target = load_credit_card_data()
data = np.array(data)
target = np.array(target)

#Creates the layers to be added to the neural network.
l1 = Layer(data.shape[1], 20,activation_function= "sigmoid")
l2 = Layer(20, 10, activation_function = "sigmoid")
l4 = Layer(10,1,activation_function = "sigmoid")

#Creates the neural network and adds the layers.
nnet = NN()
nnet.add_layer(l1)
nnet.add_layer(l2)
nnet.add_layer(l4)

#DOING A GRID SEARCH FOR FINE TUNING OF THE PARAMETERS.

n_lr_values = 5
lr_values = np.logspace(-6,-2, n_lr_values)

n_reg_lam_values =6
reg_lam_values = np.logspace(-6,-1,n_reg_lam_values)

auc_scores = np.zeros(shape=(n_lr_values, n_reg_lam_values))
accuracy_scores = np.zeros(shape=(n_lr_values, n_reg_lam_values))
area_ratio_scores = np.zeros(shape=(n_lr_values, n_reg_lam_values))

#Performing the grid search over the parameters initialized.
for i, learning_rate in enumerate(lr_values):
    for j,reg_lam in enumerate(reg_lam_values):
        cv_area_ratio, cv_auc, cv_accuracy = cv_classification_scores_nn(5, data, target, nnet, 300, 10000, learning_rate, reg_lam)
        auc_scores[i,j] = cv_auc
        accuracy_scores[i,j] = cv_accuracy
        area_ratio_scores[i,j] = cv_area_ratio


#PLOTTING HEATMAPS FOR THE GRIDSEARCH
auc_scores = pd.DataFrame(auc_scores, columns=np.round(reg_lam_values,6), index =np.round(lr_values,6) )
accuracy_scores = pd.DataFrame(accuracy_scores, columns=np.round(reg_lam_values,6), index =np.round(lr_values,6) )
area_ratio_scores = pd.DataFrame(area_ratio_scores, columns=np.round(reg_lam_values,6), index =np.round(lr_values,6) )

plt.subplot(121)
ax = sns.heatmap(area_ratio_scores, annot=True, fmt=".4f", linewidths=0.5, cbar_kws={"label": "area ratio score" })
plt.ylim(top = 0, bottom=n_lr_values)
plt.xlabel("regularization parameter")
plt.ylabel("learning rate")

plt.subplot(122)
ax = sns.heatmap(accuracy_scores, annot=True, fmt=".4f", linewidths=0.5, cbar_kws={"label": "accuracy score" })
plt.ylim(top = 0, bottom=n_lr_values)
plt.xlabel("regularization parameter")
plt.ylabel("learning rate")

plt.show()

###############################################################################################
#Found the best values of learning rate and number of iterations. lr = 0.0001, iterations = 250.
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

#Scales the input data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Uses the same arcitecture as before for the optimal parameters.
nn_best = nnet

#Trains the network with the optimal parameters.
nn_best.train(X_train, y_train, 300, 10000, alpha = 0.001, lam = 0.01)


y_probabilities = nn_best.predict(X_test)
y_pred = np.where(y_probabilities >= 0.5, 1, 0)

print("Test-accuracy: ", accuracy_score(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred)
plt.ylim(top = -0.5, bottom = 1.5)
plt.show()

#Creates new matrix with probabilities of class 0 also.
temp = np.concatenate((1-y_probabilities, y_probabilities), axis=1)

#Plots the roc curve
plot_roc(y_test, temp,plot_micro=False, plot_macro=False)
plt.show()

#Extract the curves and calculated the areas needed. 
ax = plot_cumulative_gain(y_test, temp)
lines = ax.lines
print(lines)
cm =lines[1]
print(cm.get_ydata())
plt.ylabel("Cumulative proportion of target data")
plt.xlabel("Proportion of total data")
plt.show()
