import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
from project1_lib import make_franke_vecs
from library import load_breast_cancer_data, load_credit_card_data
from numba import jit
import time

from scikitplot.metrics import plot_confusion_matrix, plot_roc, plot_cumulative_gain


class Layer:
    """Layer class. Layers can be added to instances of the NN class."""

    def __init__(self, num_input, num_nodes, activation_function = None):
        """Initializes the layer.
        Initializes the layer with weight matrix, bias vector and empty
        gradient vectors. It also sets the activation function.
        Parameters
        ----------
        num_input : int
            Number of nodes in the previous layer
        num_nodes : int
            Number of nodes in this layer
        activation_function : string, optional
            Activation function for this layer (default is None)
            Use "sigmoid", "relu" or None.
            For regression, last layer should not have sigmoid.
            For classification, last layer should have sigmoid.
        """

        self.num_input = num_input
        self.num_nodes = num_nodes

        self.weights = np.random.normal(size=(num_input, num_nodes))
        self.bias = np.random.normal(size = (1,num_nodes))

        self.activated_values = None
        self.derivative_activated_value = None

        self.activation_function = activation_function

        self.delta = None

        self.weight_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.weights.shape)

    def activate_layer(self, X):
        """Activates the layer with the approriate activation function.
        Takes the output from the previous layer, and activates each node
        in the layer.
        Parameters
        ----------
        X : numpy 2D array
            Data matrix. Samples in the rows, features in the columns.
            Dimension is (number of samples, number of features)
        Returns
        -------
        numpy 2D array
            Contains the activated values for this layer for each sample.
            Dimension is (number of samples, number of nodes in layer)
        """

        W = self.weights
        b = self.bias
        z = np.dot(X, W) + b

        if self.activation_function == "sigmoid":
            vals = 1/(1+np.exp(-z))
            self.activated_values = vals
            self.derivative_activated_values = self.activated_values*(1-self.activated_values)

        elif self.activation_function == "relu":
            vals = np.maximum(0, z)
            self.activated_values = vals
            self.derivative_activated_values = np.heaviside(z, 0.)

        elif self.activation_function == "leaky_relu":
            vals = np.maximum(0.01*z, z)
            self.activated_values = vals
            der = np.heaviside(z, 0.)
            der[np.where(der < 1)] += 0.01
            self.derivative_activated_values = der

        else:
            self.activated_values = z
            self.derivative_activated_values = 1

        return self.activated_values


class NN:
    """Class for setting up the neural network"""

    def __init__(self, cost_function = "cross_entropy"):
        """Initializes the network, and sets up an list for the layers.
        Parameters
        ----------
        cost_function : string, optional
            Specifies the cost function. (Default is "cross_entropy")
            For regression, use "mse".
            For classification, use "cross_entropy"
        """

        self.layers = []
        self.cost_function = cost_function

    def add_layer(self, layer):
        """Adds a layer to the networks list of layers.
        Parameters
        ----------
        layer : Layer
            Layer to be added to the list of layers.
        """

        self.layers.append(layer)

    def feed_forward(self, X):
        """Feeds the input forward through the network to the output layer.

        Parameters
        ----------
        X : numpy 2D array
            Data matrix. Samples in the rows, features in the columns.
            Dimension is (number of samples, number of features)

        Returns
        -------
        numpy 2D array
            The activated values of the output layer.
            Dimension is (number of samples, number of nodes in layer)
        """

        self.layers[0].activate_layer(X)

        for i in range(1, len(self.layers)):
            self.layers[i].activate_layer(self.layers[i-1].activated_values)

        return self.layers[-1].activated_values

    def predict(self, X):
        """Returns the activated values in the last layer.

        Parameters
        ----------
        X : numpy 2D array
            Data matrix. Samples in the rows, features in the columns.
            Dimension is (number of samples, number of features)

        Returns
        --------
        numpy 2D array
            The activated values of the output layer.
            Dimension is (number of samples, number of nodes in layer)
        """

        return self.feed_forward(X)

    def backpropagation(self, X, target, alpha):
        """Performs one step of backpropagation and updates each layer's
        weights and biases.

        Parameters
        ----------
        X : numpy 2D array
            Data matrix. Samples in the rows, features in the columns.
            Dimension is (number of samples, number of features)

        target : numpy 2D array.
            Target values for each sample.
            Dimension is (number of samples, 1)

        alpha : float
            Learning rate for the gradient descent method.
        """

        probability = self.feed_forward(X)

        #Updating values for the output layer
        self.layers[-1].delta = self._cost_grad(probability, target)*\
                                (self.layers[-1].derivative_activated_values)
        self.layers[-1].weight_grad = np.dot(self.layers[-2].activated_values.T,  self.layers[-1].delta)
        self.layers[-1].bias_grad = np.sum(self.layers[-1].delta,axis=0,keepdims=True)

        #Loops over the remaining layers, except the first hidden layer and
        #updates the values for each layer looped over.
        for i in range(2, len(self.layers)):
            self.layers[-i].delta = np.dot(self.layers[-i+1].delta, self.layers[-i+1].weights.T)*\
                                                (self.layers[-i].derivative_activated_values)
            self.layers[-i].weight_grad = np.dot(self.layers[-i-1].activated_values.T, self.layers[-i].delta)
            self.layers[-i].bias_grad = np.sum(self.layers[-i].delta, axis=0, keepdims=True)

        #Updating the values for the first hidden layer.
        self.layers[0].delta = np.dot(self.layers[1].delta, self.layers[1].weights.T)
        self.layers[0].weight_grad = np.dot(X.T, self.layers[0].delta)
        self.layers[0].bias_grad = np.sum(    self.layers[0].delta, axis=0, keepdims=True)

        #Updates the layers weights with the gradients with gradient descent.
        for layer in self.layers:
            layer.weights -= alpha*(layer.weight_grad + self.lam*layer.weights)
            layer.bias -= alpha*(layer.bias_grad + self.lam*layer.bias)


    def train(self, X, target, epochs, batch_size, alpha, lam,
              plot_cost_over_epoch = False, verbose = False):
        """Trains the network using SGD.

        Parameters
        -----------
        X : numpy 2D array,
            Data matrix. Samples in the rows, features in the columns.
            shape(number of samples, number of features)

        target : numpy 2D array.
            Target values for each sample.
            shape(number of samples, 1)

        epochs : int
            Number of epochs for training the network.

        batch_size : int
            Batch size for the stochastic gradient descent method.

        alpha : float
            Learning rate for the stochastic gradient descent.

        plot_cost_over_epoch: bool
            If plots should be presented after training.

        verbose: bool
            If information should be printed during training.

        Returns
        -------
        cost_over_epochs: numpy vector
            Vector with cost function value as a function of epochs in training.
        """
        self.lam = lam

        for layer in self.layers:
            layer.weights = np.random.normal(size=(layer.num_input, layer.num_nodes))
            layer.bias = np.random.normal(size = (1,layer.num_nodes))

        num_samples = X.shape[0]
        data_indices = np.arange(num_samples)
        iterations = num_samples // batch_size

        cost_over_epochs = []

        start_time = time.time()
        for i in range(epochs):
            for j in range(iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=batch_size, replace=False
                )

                X_batch = X[chosen_datapoints]
                target_batch = target[chosen_datapoints]

                self.backpropagation(X_batch, target_batch, alpha)

            # if i % 10 == 0:
            #     if self.cost_function == "mse":
            #         print("train MSE = ", self._cost(self.predict(X), target))
            #         cost_over_epochs.append(self._cost(self.predict(X), target))
            #         # print("progress:", "|"*i/10., ","*(epochs - i)/10.)
            #     elif self.cost_function == "cross_entropy":
            #         print("Cross-entropy = ", self._cost(self.predict(X), target))
            #         cost_over_epochs.append(self._cost(self.predict(X), target))
            #         # print("progress:", "|"*i, ","*(epochs - i))
            cost_over_epochs.append(self._cost(self.predict(X), target))
            if i % 10 == 0 and verbose:
                print("Epoch:", i, "   %s: " %(self.cost_function), "%.5f" %self._cost(self.predict(X), target), "  Time elapsed: %.2f" %(time.time()-start_time), "s")

        if plot_cost_over_epoch:
            plt.plot(cost_over_epochs)
            plt.show()
        return cost_over_epochs



    def _cost(self, probability, target):
        if self.cost_function == "cross_entropy":
            return -np.sum(target * np.log(probability) + \
                                    (1-target)*np.log(1-probability))

        elif self.cost_function == "mse":
            return np.mean((probability - target)**2)


    def _cost_grad(self, probability, target):
        if self.cost_function == "cross_entropy":
            return   probability - target

        elif self.cost_function == "mse":
            return probability - target



def cv_classification_scores_nn(num_splits,
                                data,
                                targets,
                                nnet,
                                epochs,
                                batch_size,
                                alpha,
                                lam):
    """Calculates the Cross-validation AUC-score and accuracy-score for classification neural nets.

    Function for calculating the cross-validation scores. nnet have to be an instance
    of the NN class.

    Parameters
    ----------
    num_splits: int
        Number of cross validation splits/folds.

    data: numpy 2d array
        Matrix containing the data to fit and test the model.

    targets: numpy 2d array
        Vector containing the target values for the data matrix.

    nnet: NN
        Neural network object.

    epochs: int
        Number of training epochs in the trianing of the network.

    batch_size: int
        Size of each batch in the stochastic gradient descent in the training of
        the network.

    alpha: float
        Learning rate for the gradient descent method in the training of the
        network.

    lam: float
        Regularization parameter.

    Returns
    -------
    cv_auc: float
        Cross-validation score for the AUC value on data.

    cv_accuracy: float
        Cross-validation score for the accuracy value on data.

    """

    k_fold = KFold(n_splits = num_splits, shuffle=True)

    cv_auc_scores = np.zeros(num_splits)
    cv_accuracy_scores = np.zeros(num_splits)
    cv_area_ratio_score = np.zeros(num_splits)

    i = 0

    for train_index, test_index in k_fold.split(data):
        sc = StandardScaler()

        x_train = data[train_index]
        x_test = data[test_index]

        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        y_train = targets[train_index]
        y_test = targets[test_index]

        nnet.train(x_train, y_train, epochs, batch_size, alpha, lam,
                            plot_cost_over_epoch = False, verbose = False)

        proba = nnet.predict(x_test)

        temp = np.concatenate((1-proba, proba), axis=1)

        ax = plot_cumulative_gain(y_test, temp)
        plt.close()
        lines = ax.lines
        cm =lines[1]
        model_ydata = cm.get_ydata()
        model_xdata = cm.get_xdata()

        area_model_curve = np.trapz(model_ydata, model_xdata)

        area_optimal_curve = 0.788 + 0.5*0.2212

        area_baseline_curve = 0.5

        area_ratio = (area_model_curve-area_baseline_curve)/(area_optimal_curve - area_baseline_curve)

        cv_area_ratio_score[i] = area_ratio
        print("Area-ratio" , cv_area_ratio_score[i])

        cv_auc_scores[i] = roc_auc_score(y_test, proba)
        print("AUC-score: ", roc_auc_score( y_test, proba))

        predicted = np.where(proba >= 0.5, 1, 0)
        cv_accuracy_scores[i] = accuracy_score(y_test, predicted)

        print("Accuracy: ", accuracy_score(y_test, predicted))
        print(confusion_matrix(y_test, predicted))

        i+=1

    cv_auc = np.mean(cv_auc_scores)
    cv_accuracy = np.mean(cv_accuracy_scores)
    cv_area_ratio = np.mean(cv_area_ratio_score)

    print("CV-area ratio: ", cv_area_ratio, "CV-AUC: ", cv_auc, "  CV-accuracy: ", cv_accuracy )
    return cv_area_ratio, cv_auc, cv_accuracy

def cv_regression_scores_nn(num_splits,
                            data,
                            targets,
                            nnet,
                            epochs,
                            batch_size,
                            alpha,
                            lam):
    """Calculates the Cross-validation AUC-score and accuracy-score for regression neural nets.

    Function for calculating the cross-validation scores. nnet have to be an instance
    of the NN class.

    Parameters
    ----------
    num_splits: int
        Number of cross validation splits/folds.

    data: numpy 2d array
        Matrix containing the data to fit and test the model.

    targets: numpy 2d array
        Vector containing the target values for the data matrix.

    nnet: NN
        Neural network object.

    epochs: int
        Number of training epochs in the trianing of the network.

    batch_size: int
        Size of each batch in the stochastic gradient descent in the training of
        the network.

    alpha: float
        Learning rate for the gradient descent method in the training of the
        network.

    lam: float
        Regularization parameter.

    Returns
    -------
    cv_mse: float
        Cross-validation score for the mse value on data.

    """

    k_fold = KFold(n_splits = num_splits, shuffle=True)

    cv_mse_scores = np.zeros(num_splits)


    i = 0

    for train_index, test_index in k_fold.split(data):
        sc = StandardScaler()

        x_train = data[train_index]
        x_test = data[test_index]

        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        y_train = targets[train_index]
        y_test = targets[test_index]

        nnet.train(x_train, y_train, epochs, batch_size, alpha, lam,
                            plot_cost_over_epoch = False, verbose = False)

        predicted = nnet.predict(x_test)

        cv_mse_scores[i] = mean_squared_error(y_test, predicted)

        print(i)
        print("MSE_final: ", cv_mse_scores[i])

        i+=1

    cv_mse = np.mean(cv_mse_scores)

    print("CV-MSE: ", cv_mse)
    return cv_mse

def plot_bias_variance_nn_reg(data, response, num_lambda_vals, learning_rate):
    """
    Function for plotting the bias-variance trade-off using Ridge regression.
    Plots for different values of lamba between 1e-3 to 1e1.
    """
    num_splits = 5

    bias_values = np.zeros(num_lambda_vals)
    variance = np.zeros(num_lambda_vals)
    MSE_vals = np.zeros(num_lambda_vals)
    max_exp = -1
    min_exp = -6

    x_vals = np.logspace(min_exp,max_exp, num_lambda_vals)

    X_train_1, X_test, z_train_1, z_test = train_test_split(data, response, test_size = 0.2)#,random_state = 42)

    l = len(z_test)

    k = 0

    for i in np.logspace(min_exp,  max_exp, num_lambda_vals):#range(0, max_lambda):
        k_folds = KFold(n_splits = num_splits, shuffle=True, random_state=1)

        fold_score = np.zeros(num_splits)

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


        predictions = np.zeros((l, num_splits))

        j = 0

        for train_index, test_index in k_folds.split(X_train_1):
            print(train_index)
            # X_train_2 = data_matrix_train[train_index]
            # # X_val = X_train_1[test_index]
            # z_train_2 = z_train_1[train_index]



            # z_val = z_train[test_index]

            nnet.train(X_train_1, z_train_1, 500, 50, learning_rate, i )

            y_pred = nnet.predict(X_test)
            # print(y_pred)

            predictions[:,j] = y_pred.T

            # fold_score[i] = MSE(y_pred, y_test)

            j+=1
        bias_values[k] = np.mean((z_test - np.mean(predictions, axis = 1))**2)
        variance[k] = np.mean(np.var(predictions, axis = 1))
        MSE_vals[k] = np.mean( np.mean((z_test[:,np.newaxis] - predictions)**2, axis=1, keepdims=True) )

        k+=1
    plt.plot(x_vals, variance, linestyle="--",marker="o")
    plt.plot(x_vals, bias_values, linestyle ="--", marker="o")
    plt.plot(x_vals, MSE_vals, linestyle="--", marker="o")
    plt.xscale("log")
    # plt.logplot(x_vals, variance, linestyle="--",marker="o")
    # plt.logplot(x_vals, bias_values, linestyle ="--", marker="o")
    # plt.logplot(x_vals, MSE_vals, linestyle="--", marker="o")
    plt.legend(["variance", "bias", "MSE"])
    plt.xlabel("regularization parameter")
    plt.title("Bias-variance neural network, learning rate = %.6f" %(learning_rate))
    print("min MSE:", min(MSE_vals))




"""
CLASSIFICATION START
"""

# # data, target = load_breast_cancer_data()
# data, target = load_credit_card_data()
# sc = StandardScaler()
# data = sc.fit_transform(data)
#
# l1 = Layer(data.shape[1], 6,activation_function= "sigmoid")
# # l2 = Layer(6, 3, activation_function = "sigmoid")
# l3 = Layer(6,1,activation_function = "sigmoid")
#
# nnet = NN()
# nnet.add_layer(l1)
# # nnet.add_layer(l2)
# nnet.add_layer(l3)
#
# nnet.train(data, target, 300, 10000, 0.00001)
#
# pred = np.where(nnet.predict(data) >= 0.5, 1, 0)
#
# print(accuracy_score(pred, target))
# print(confusion_matrix(target, pred))

"""
CLASSIFICATION END
"""

"""
REGRESSION START
"""
# data, target = make_franke_vecs(0.1, 100)
# target = np.expand_dims(target, axis=1)
# print(data.shape, target.shape)
#
# l1 = Layer(data.shape[1], 5,activation_function= "leaky_relu")
# l2 = Layer(5, 3, activation_function = "leaky_relu")
# l3 = Layer(3,2,activation_function = "leaky_relu")
# l4 = Layer(2,1,activation_function = None)
#
# nnet = NN(cost_function="mse")
# nnet.add_layer(l1)
# nnet.add_layer(l2)
# nnet.add_layer(l3)
# nnet.add_layer(l4)
#
# nnet.train(data, target, 100, 500, 0.001)
# pred = nnet.predict(data)
# print(pred.shape)
# print(np.mean((nnet.predict(data) - target)**2))

"""
REGRESSION END
"""
