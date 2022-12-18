# %%
# Importing libraries
# TODO: clean up imports
import matplotlib.pyplot as plt
import numpy as np
from random import seed
# import numba as nb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.utils import resample



def MSE(y_data, y_model):
    """Computes mean squared error."""
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n


def create_X_polynomial(x, y, n):
    """Computes the design matrix for a degree n polynomial in variables
    x and y."""
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y ** k)

    return X


class LinearRegression:

    def __init__(self, modeltype="ols"):
        """Initialize the model type and set lambda and max iterations parameter
        if required.
        
        Parameters
        ----------
        modeltype: str
            must be 'ols'
        max_iter: 
        """
        modeltype = modeltype.lower()
        if modeltype not in ["ols"]:
            raise ValueError("Model must be ols")
        
        self._model = modeltype
    
    def __call__(self, X, z):
        """Estimates beta parameter using stored model type.

        Parameters
        ----------
        X: np.ndarray
            Design matrix
        z: np.ndarray
            Dependent variable
        
        Returns
        -------
        beta_hat: np.ndarray
            Estimated beta parameters
        """
        if self._model == "ols":
            beta_hat = ols_regression(X, z)
        else:
            raise RuntimeError("Could not find model")
        
        return beta_hat
    def __str__(self):
        name = {
            "ols": "Ordinary Least Squares Regression", 
            }
        return name[self._model]


def ols_regression(X, z):
    """Performs Ordinary Least Squares regression to estimate
    beta parameters."""
    # Solving for beta
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z
    return beta




def bootstrap(x, y, z, deg, model, B, test_size=0.25):
    """Returns estimated distributions of beta estimators for OLS
    
    Parameters
    ----------
    model: LinearRegression object
        Must be ols
    x: np.ndarray
        x-coordinates
    y: np.ndarray
        y-coordinates
    z: np.ndarray
        Dependent variable
    deg: int
        Polynomial degree used for regression
    model: LinearRegression
        Linear regression model used, either ols, ridge, or lasso
    B: int
        Number of bootstrap iterations
    test_size: float
        Proportion of data used as test set

    Returns
    -------
    bias: float
        Estimated bias
    variance: float
        Estimated variance
    error: float
        Test set error
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    X = create_X_polynomial(x, y, deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=test_size)
    z_train = z_train.reshape(-1, 1)
    z_test = z_test.reshape(-1, 1)

    z_pred = np.empty((z_test.shape[0], B))
    for i in range(B):
        X_, z_ = resample(X_train, z_train)
        z_pred[:,i] = X_test @ (model(X_, z_)).ravel()


    bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2)
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True))
    error = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    return bias, variance, error


def bootstrap_tree(X_train, X_test, z_train, z_test, model, B):
    """Returns estimated distributions of beta estimators for a decision tree regressor."""
    z_pred = np.empty((z_test.shape[0], B))
    for i in range(B):
        X_, z_ = resample(X_train, z_train)
        model.fit(X_, z_)
        z_pred[:,i] = model.predict(X_test)


    bias = []
    variance = []
    mse = []

    for i in range(len(z_test)):
        bias.append((z_test[i] - np.mean(z_pred[i]))**2)
        variance.append(np.var(z_pred[i]))
        mse.append(np.mean((z_test[i] - z_pred[i])**2))

    # Find the bias and variance for neural network
    bias = np.mean(bias)
    variance = np.mean(variance)
    error = np.mean(mse)

    return bias, variance, error

def bootstrap_nn(X_train, X_test, z_train, z_test, model, B):
    """Returns estimated distributions of beta estimators for a decision tree regressor."""
    z_pred = np.empty((z_test.shape[0], B))
    for i in range(B):
        X_, z_ = resample(X_train, z_train)
        model.fit(X_, z_)
        z_pred[:,i] = model.predict(X_test).ravel()

    bias = []
    variance = []
    mse = []

    for i in range(len(z_test)):
        bias.append((z_test[i] - np.mean(z_pred[i]))**2)
        variance.append(np.var(z_pred[i]))
        mse.append(np.mean((z_test[i] - z_pred[i])**2))

    # Find the bias and variance for neural network
    bias = np.mean(bias)
    variance = np.mean(variance)
    error = np.mean(mse)

    return bias, variance, error


    