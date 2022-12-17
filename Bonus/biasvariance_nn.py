import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
# Import neural network from scikit-learn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split




x, y, z, _ = dg.generate_data_Franke(500, 0.1, seed=1)
X = np.array([[x[i], y[i]] for i in range(len(x))])
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)



max_depth_range = 25
bias_arr = []
variance_arr = []
error_arr = []
progress = 0
print(f"Progress: {progress} %", end="\r")

x_vals = []

for i in range(1, max_depth_range, 2):
    x_vals.append(i)
    clf = MLPRegressor(hidden_layer_sizes=(i,), activation='logistic', solver='lbfgs', max_iter=2000)
    bias, variance, error = rt.bootstrap_nn(X_train, X_test, z_train, z_test, clf, 100)
    bias_arr.append(bias)
    variance_arr.append(variance)
    error_arr.append(error)
    progress = int(i/max_depth_range*100)
    print(f"Progress: {progress} %", end="\r")



plt.plot(x_vals, bias_arr, "--o", label='Bias^2', color='blue')
plt.plot(x_vals, variance_arr, "--o", label='Variance', color='orange')
plt.plot(x_vals, error_arr, "--o", label='Error', color='green')
plt.title("Bias-variance tradeoff for neural network")
plt.xlabel('Number of nodes in hidden layer')
plt.ylabel('Error scores')
plt.legend()
plt.show()
