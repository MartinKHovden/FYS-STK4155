from nn import Layer, NN
import numpy as np
import random

#TESTING OF THE MAIN FUNCTIONS OF THE NEURAL NETWORK CLASS.

def test_Layer():
    """Tests the layer class and checks if the activate layer function
    works.
    """

    l1 = Layer(2, 2, "relu")

    l1.weights = np.array([[0.1, 0.1],[0.2,0.2]])
    l1.bias = np.array([[0.1, 0.1]])

    input = np.array([[1,1],[1,1]])

    out = np.array(l1.activate_layer(input))
    true_out = np.array([[0.4, 0.4], [0.4, 0.4]])

    assert(np.allclose(out,true_out))

def test_NN_feed_forward():
    """Test the feed forward function of the neural network class.

    """

    l1 = Layer(2, 2, "relu")
    l2 = Layer(2,1)

    l1.weights = np.array([[0.1, 0.1],[0.2,0.2]])
    l1.bias = np.array([[0.1, 0.1]])
    l2.weights = np.array([[0.2], [0.2]])
    l2.bias = np.array([0.1])

    input = np.array([[1,1],[1,1]])
    nnet = NN()

    nnet.add_layer(l1)
    nnet.add_layer(l2)

    out = nnet.predict(input)
    expected_out = np.array([[0.26], [0.26]])

    assert(np.allclose(out, expected_out))

def test_backpropagation():
    """Tests the backpropagation algorithm in the neural network class
    """

    l1 = Layer(2, 2, "relu")
    l2 = Layer(2,1)

    l1.weights = np.array([[0.1, 0.1],[0.2,0.2]])
    l1.bias = np.array([[0.1, 0.1]])
    l2.weights = np.array([[0.2], [0.2]])
    l2.bias = np.array([[0.1]])

    input = np.array([[1,1],[1,1]])
    nnet = NN()

    nnet.add_layer(l1)
    nnet.add_layer(l2)

    nnet.lam = 0

    target = np.array([[1],[1]])

    nnet.backpropagation(input, target, 1)
    weight_grad = nnet.layers[0].weight_grad
    bias_grad = nnet.layers[0].bias_grad
    expected_weight_grad = np.array([[-0.296, -0.296], [-0.296, -0.296]])
    expected_bias_grad = np.array([[-0.296, -0.296]])

    assert(np.allclose(expected_weight_grad, weight_grad) and np.allclose(expected_bias_grad, bias_grad))

#TESTING OF THE MAIN FUNCTIONS IN THE LOGISTIC REGRESSION CLASS.

from library import LogReg, load_breast_cancer_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def test_logreg_sigmoid():
    """Tests the sigmoid function implemented in the logreg class
    """

    logreg = LogReg(10)
    assert(np.allclose(logreg.sigmoid(0), 0.5 )and np.allclose(logreg.sigmoid(1), 0.731058578 ) \
                    and np.allclose(logreg.sigmoid(np.array([[0],[0]])), np.array([[0.5], [0.5]])) )

def test_logreg_probabilities():
    """Test to see if the logreg class returns the correct probabilities
    for a given input.
    """

    logreg = LogReg(10)
    logreg.beta_values = np.array([1,1, -2])
    input = np.array([[1,1],[1,1]])
    proba1 = logreg.probabilities(input)
    expected_proba1 = [0.5, 0.5]

    logreg.beta_values = np.array([1,1, -1])
    input = np.array([[1,2],[1,2]])
    proba2 = logreg.probabilities(input)
    expected_proba2 = [0.5, 0.5]

    assert(np.allclose(proba1, expected_proba1) and np.allclose(proba2, expected_proba2))

def test_logreg_fit():
    """Test if the logreg class' fit function gives the correct beta values.
    """
    logreg = LogReg(10)
    logreg.beta_values = np.array([1,1, -2])
    input = np.array([[1,1],[1,1]])
    y = np.array([[1],[1]])
    logreg.fit(input, y)
    expected = np.array([[0.00099933],[0.00099933], [0.00099933]])
    betas = logreg.beta_values

    assert(np.allclose(betas, expected))
