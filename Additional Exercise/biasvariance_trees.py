import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine as wine


# Generate and split data
x, y, z, _ = dg.generate_data_Franke(500, 0.1, seed=1)
X = np.array([[x[i], y[i]] for i in range(len(x))])
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)




# Set up arrays for plotting
max_depth_range = 15
bias_arr = []
variance_arr = []
error_arr = []
x_vals = []
print(f"Progress: {0} %", end="\r")

# Loop over different tree depths
for i in range(1, max_depth_range):
    x_vals.append(i)
    #print("---------------Depth: ", i, "------------------")
    clf = tree.DecisionTreeRegressor(max_depth=i)
    bias, variance, error = rt.bootstrap_tree(X_train, X_test, z_train, z_test, clf, B=100)
    bias_arr.append(bias)
    variance_arr.append(variance)
    error_arr.append(error)
    print(f"Progress: {int(i/max_depth_range*100)} %", end="\r")


# Plot results
plt.ylim(0, 0.05)
plt.plot(x_vals, bias_arr, "--o", label='Bias^2', color='blue')
plt.plot(x_vals, variance_arr, "--o", label='Variance', color='orange')
plt.plot(x_vals, error_arr, "--o", label='Error', color='green')
plt.title("Bias-variance tradeoff for decision tree regressor")
plt.xlabel('Tree depth')
plt.ylabel('Error scores')
plt.legend()
plt.show()
