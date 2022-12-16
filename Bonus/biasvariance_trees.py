import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine as wine



x, y, z, _ = dg.generate_data_Franke(1000, 0.1, seed=1)
X = np.array([[x[i], y[i]] for i in range(len(x))])
X_train, X_test, z_train, z_test = train_test_split(X, y, test_size=0.2)




"""
# Load data
data = win()
X = data.data
y = data.target

# Split data into training and test sets
X_TR, X_test, z_TR, z_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, z_train, z_test = train_test_split(X_TR, z_TR, test_size=0.2)
n = X_train.shape[1]
"""


max_depth_range = 1000
bias_arr = []
variance_arr = []

for i in [10]:
    clf = tree.DecisionTreeRegressor(max_depth=i)
    bias, variance = rt.bootstrap_tree(X_train, X_test, z_train, z_test, clf, 100)
    plt.hist(bias.flatten(), bins=30)
    plt.show()
    plt.hist(variance, bins=30)
    plt.show()


plt.plot(np.linspace(1, max_depth_range, max_depth_range-1), bias_arr, label='Bias')
plt.plot(np.linspace(1, max_depth_range, max_depth_range-1), variance_arr, label='Variance')
plt.xlabel('Max depth')
plt.ylabel('Error')
plt.legend()
plt.show()

