import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
from sklearn import tree
from sklearn.model_selection import train_test_split



# Generate and split data
x, y, z, _ = dg.generate_data_Franke(500, 0.1, seed=1)
X = np.array([[x[i], y[i]] for i in range(len(x))])
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.25)



# Set up arrays for plotting
ols = rt.LinearRegression("ols")
degreerange = 12
degrees = list(range(1, degreerange + 1))
biases, variances, errors = [], [], []


# Loop over different polynomial degrees
for deg in degrees:
    bias, variance, error = rt.bootstrap(x, y, z, deg, ols, B=100)
    biases.append(bias)
    variances.append(variance)
    errors.append(error)

# Plot results
plt.ylim(0, 0.05)
plt.plot(degrees, biases, "--o" , label="Bias^2")
plt.plot(degrees, variances, "--o" ,label="Variance")
plt.plot(degrees, errors, "--o", label="Error")
plt.title("Bias-variance tradeoff for OLS")
plt.xlabel("Polynomial order")
plt.ylabel("Error scores")
plt.legend()
plt.show()
plt.clf()