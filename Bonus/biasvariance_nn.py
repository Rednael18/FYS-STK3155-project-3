import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
# Import neural network from scikit-learn
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split




x, y, z, _ = dg.generate_data_Franke(20, 0.1, seed=1)
X = np.array([[x[i], y[i]] for i in range(len(x))])
X_train, X_test, z_train, z_test = train_test_split(X, y, test_size=0.2)



max_depth_range = 15
bias_arr = []
variance_arr = []
error_arr = []
progress = 0
print(f"Progress: {progress} %", end="\r")

for i in range(2, max_depth_range):
    clf = MLPRegressor(hidden_layer_sizes=(i, i,), activation='logistic', solver='lbfgs', max_iter=1000)
    bias, variance, error = rt.bootstrap_nn(X_train, X_test, z_train, z_test, clf, 100)
    bias_arr.append(bias)
    variance_arr.append(variance)
    error_arr.append(error)
    progress = int(i/max_depth_range*100)
    print(f"Progress: {progress} %", end="\r")



plt.plot(np.linspace(2, max_depth_range, max_depth_range-2), bias_arr, label='Bias')
plt.plot(np.linspace(2, max_depth_range, max_depth_range-2), variance_arr, label='Variance')
plt.plot(np.linspace(2, max_depth_range, max_depth_range-2), error_arr, label='Error')
plt.xlabel('Max depth')
plt.ylabel('Error')
plt.legend()
plt.show()
