# FYS-STK4155 PROJECT 2: Neural networks for classification and regression.

## General info
The code and report is based on project 2 in the course FYS-STK4155. The goal was use neural network for classification
and regression, and compare the neural network to other popular machine learning
algorithms. The data sets used are the credit card data from UCI and the Franke function.
The results are presented in the report together with a discussion about the theory and methods.

The main functions and classes can be found in library.py and nn.py. The classes needed for building
a neural network from scratch are in nn.py. Here we also find functions for calculating k-fold cross
validation for different performance measures. In the library.py file, function for loading the
data into matrices and vectors are found. In addition, a class for building logistic regression are there.
Function for calculating k-fold cross validation are also there.

The plots and results are created using the codes in the files b.py, c.py and d.py. A corresponding jupyter
notebook for each part is also in the repository. However, the code in the python scripts might be a bit
better commented. The problem with them is that you have to comment out some of the code to avoid running every calculations (some takes over an hour). the jupyter notebooks solves this problem, and should be ok to follow as well.

In the jupyter notebooks credit_card_exploration.jpynb and comparion.jpynb code for exploration, cleaning and comparison of the methods to sklearn are found. I decided to not include any of the comparisons with sklearn in the project report.

Plots/benchmarks are found in the folder Plots. Plots don't have titles to make them nicer in the report but they are named according to the problem they solve. In the project report you can find most of the plots with a better explanation about what the plot shows.

## Testing
To run tests on the main functions of the neural network and logistic regression:
`pytest test.py`.
The test.py file includes basic unit tests to see if the main part of the classes for neural network and logistic regression works as it should. The tests are non-deterministic and fixed weights and biases are given to the models.  
