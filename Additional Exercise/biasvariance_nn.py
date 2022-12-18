import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
# Import neural network from scikit-learn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split



# Generate and split data
x, y, z, _ = dg.generate_data_Franke(500, 0.1, seed=1)
X = np.array([[x[i], y[i]] for i in range(len(x))])
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)


# Set up arrays for plotting
max_node_range = 50
bias_arr = []
variance_arr = []
error_arr = []
progress = 0
x_vals = []
print(f"Progress: {progress} %", end="\r")


# Loop over different number of nodes in hidden layer
for i in range(1, max_node_range, 2):
    x_vals.append(i)
    clf = MLPRegressor(hidden_layer_sizes=(i,), activation='logistic', solver='lbfgs', max_iter=2000)
    bias, variance, error = rt.bootstrap_nn(X_train, X_test, z_train, z_test, clf, 100)
    bias_arr.append(bias)
    variance_arr.append(variance)
    error_arr.append(error)
    progress = int(i/max_node_range*100)
    print(f"Progress: {progress} %", end="\r")


# Plot results
plt.ylim(0, 0.05)
plt.plot(x_vals, bias_arr, "--o", label='Bias^2', color='blue')
plt.plot(x_vals, variance_arr, "--o", label='Variance', color='orange')
plt.plot(x_vals, error_arr, "--o", label='Error', color='green')
plt.title("Bias-variance tradeoff for neural network")
plt.xlabel('Number of nodes in hidden layer')
plt.ylabel('Error scores')
plt.legend()
plt.show()
