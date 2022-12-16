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


def R2(y_data, y_model):
    """Computes R^2-score."""
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def MSE(y_data, y_model):
    """Computes mean squared error."""
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n

def accuracy(y_data, y_model):
    """Computes the accuracy of the model."""
    y_model = y_model.flatten()
    return np.sum(y_data == y_model) / len(y_data)


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

    def __init__(self, modeltype="ols", lambdan=None):
        """Initialize the model type and set lambda and max iterations parameter
        if required.
        
        Parameters
        ----------
        modeltype: str
            must be either 
                'ols'   (for Ordinary Least Squares regression)
                'ridge' (for Ridge regression)
                'lasso' (for Lasso regression)
        lambdan: None /float
            For ols, this parameter is None
            For ridgeg/lasso this parameter is in [0, +inf)
        max_iter: 
        """
        modeltype = modeltype.lower()
        if modeltype not in ["ols", "ridge", "lasso"]:
            raise ValueError("Model must be ols/ridge/lasso")
        
        if modeltype == "ols":
            if lambdan != None:
                raise ValueError("ols model has no parameter lambdan")
        elif modeltype in ["ridge", "lasso"]:
            if lambdan == None:
                raise ValueError(f"{modeltype} requires parameter lambdan")
        
        self._model = modeltype
        self._lambda = lambdan
    
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
        elif self._model == "ridge":
            beta_hat = ridge_regression(X, z, self._lambda)
        elif self._model == "lasso":
            beta_hat = lasso_regression(X, z, self._lambda)
        else:
            raise RuntimeError("Could not find model")
        
        return beta_hat
    def __str__(self):
        name = {
            "ols": "Ordinary Least Squares Regression", 
            "ridge": "Ridge Regression", 
            "lasso": "Lasso Regression"
            }
        return name[self._model]


def ols_regression(X, z):
    """Performs Ordinary Least Squares regression to estimate
    beta parameters."""
    # Solving for beta
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z
    return beta


def ridge_regression(X, z, lambdan):
    """Performs Ridge regression to estimate beta parameters."""
    # create identity matrix
    I = np.eye(len(X.T), len(X.T))

    # Solving for beta
    beta_ridge = np.linalg.pinv(X.T @ X + lambdan * I) @ X.T @ z
    return beta_ridge


# Lassso with scikit-learn:
def lasso_regression(X, z, lambdan):
    """Performs Lasso regression to estimate beta parameters."""
    # Lasso regression with scikit-learn
    RegLasso = Lasso(lambdan, fit_intercept=False)
    fit = RegLasso.fit(X, z)
    beta_coef = fit.coef_
    return beta_coef


def bootstrap(X, z, B, model):
    """Returns estimated distributions of beta estimators.
    
    Parameters
    ----------
    model: LinearRegression object
        Either OLS, Ridge, or Lasso regression
    X: np.ndarray
        Design matrix
    z: np.ndarray
        Dependent variable
    B: int
        Number of bootstrap iterations

    Returns
    -------
    distribution: np.ndarray
        An array of dimensions (len(beta), B) where distribution[:, b]
        are the parameters beta estimated at the b'th bootstrap iteration
    """
    t = np.zeros(B)
    n_datapoints = len(z)

    beta = model(X, z)
    distribution = np.zeros((len(beta), B))
    for b in range(B):
        datapoints = np.random.randint(0, n_datapoints, n_datapoints)
        X_b = X[datapoints]
        z_b = z[datapoints]
        beta_b = model(X_b, z_b)
        distribution[:, b] = beta_b
    return distribution

def bootstrap(x, y, z, deg, model, B, test_size=0.25):
    """Returns estimated distributions of beta estimators.
    
    Parameters
    ----------
    model: LinearRegression object
        Either OLS, Ridge, or Lasso regression
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


def bootstrap_tree(X_train, X_test, z_train, z_test, model, B, test_size=0.25):
    """Returns estimated distributions of beta estimators for a decision tree regressor."""
    z_pred = np.empty((z_test.shape[0], B))
    for i in range(B):
        X_, z_ = resample(X_train, z_train)
        model.fit(X_, z_)
        z_pred[:,i] = model.predict(X_test)

    bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2)
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True))
    #error = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    return bias, variance

def bootstrap_nn(X_train, X_test, z_train, z_test, model, B, test_size=0.25):
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
        variance.append(np.var(z_pred[i])/np.mean(bias))
        mse.append(np.mean((z_test[i] - z_pred[i])**2))

    # Find the bias and variance for neural network
    bias = np.mean(bias)
    variance = np.mean(variance)
    error = np.mean(mse)

    return bias, variance, error


def cross_validation(X, z, k_deg_fold, model):
    "Preformes Cross-Validation with model=linreg, X=design-matrix"
    "Returns arrays - MSE-train, MSE-test - with length k_deg_fold"
    "np.mean() on output is estimated MSE with cross-validation"

    # step 1: shuffle datasets randomly using np.random.permutation(len(x)):
    assert len(X) == len(z)
    p = np.random.permutation(len(X))
    X, z = X[p], z[p]

    # step 2: split the data in k groups with numpy.array_split
    X = np.array_split(X, k_deg_fold)
    z = np.array_split(z, k_deg_fold)

    # array to keep track of MSE for each test-group and train-group
    MSE_train = np.zeros((k_deg_fold))
    MSE_test = np.zeros((k_deg_fold))

    # step 3: for i in range of folds preform:
    for i in range(k_deg_fold):
        # a) pick one group to be test data
        X_test, z_test = X[i], z[i]

        # b) take remaining groups as train data
        # np.delete() creates a "new" array (does not alter X)
        # concatenate merges remaining groups to train data
        X_train = np.concatenate([m for m in np.delete(X, i, axis=0)])
        z_train = np.concatenate([m for m in np.delete(z, i, axis=0)])

        # c) fit model to train data with linreg and compute z_tilde
        beta = model(X_train, z_train)
        z_tilde_test = X_test @ beta
        z_tilde_train = X_train @ beta

        # d) evaluate model and save score-value to MSE-arrays
        MSE_train[i] = MSE(z_train, z_tilde_train)
        MSE_test[i] = MSE(z_test, z_tilde_test)

    return MSE_train, MSE_test


if __name__ == "__main__":
    """
    # Generating data
    from analysis import generate_data_Franke
    #seed = 2018
    N = 100
    sigma2 = 0

    x, y, z, _ = generate_data_Franke(N, sigma2, seed)
    

    # Fit an n-degree polynomial
    n = 5
    X = create_X_polynomial(x, y, n)
    model = LinearRegression("lasso", 1)
    
    
    beta = model(X, z)
    
    ztilde = X @ beta
    
    plt.scatter(x,z, c='b')
    plt.scatter(x, ztilde, c='r')
    plt.show()
    
    B = 400
    distribution = bootstrap(X, z, B, model)
    
    # Plots estimated distribution for the i'th parameter of the model
    i = 1
    print(max(distribution[i, :]))
    plt.hist(distribution[i, :])
    plt.show()
    
    k_fold = 8
    lambdan = 0.0001
    MSE_train, MSE_test = cross_validation(X, z, k_fold, model)
    print(MSE_train, MSE_test)
    #print(MSE_arr)
    """
    
    