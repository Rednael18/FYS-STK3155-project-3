import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine as wine



x, y, z, _ = dg.generate_data_Franke(500, 0.1, seed=1)
X = np.array([[x[i], y[i]] for i in range(len(x))])
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)





max_depth_range = 15
bias_arr = []
variance_arr = []
error_arr = []
print(f"Progress: {0} %", end="\r")
x_vals = []

for i in range(1, max_depth_range):
    x_vals.append(i)
    #print("---------------Node size: ", i, "------------------")
    clf = tree.DecisionTreeRegressor(max_depth=i)
    bias, variance, error = rt.bootstrap_tree(X_train, X_test, z_train, z_test, clf, 100)
    bias_arr.append(bias)
    variance_arr.append(variance)
    error_arr.append(error)
    print(f"Progress: {int(i/max_depth_range*100)} %", end="\r")


plt.plot(x_vals, bias_arr, "--o", label='Bias^2', color='blue')
plt.plot(x_vals, variance_arr, "--o", label='Variance', color='orange')
plt.plot(x_vals, error_arr, "--o", label='Error', color='green')
plt.title("Bias-variance tradeoff for decision tree regressor")
plt.xlabel('Number of nodes in hidden layer')
plt.ylabel('Error scores')
plt.legend()
plt.show()
