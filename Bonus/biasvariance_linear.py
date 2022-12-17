import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
from sklearn import tree
from sklearn.model_selection import train_test_split


print("Problem c)")

# Set parameters
N = 400
B = 100 # n_bootsraps 
sigma2 = 0.1 # Variance of noise
seed = 2340

# Generate data
x, y, z, _ = dg.generate_data_Franke(N, sigma2, seed)

# Initialize model
ols = rt.LinearRegression("ols")

# Plot train & test MSE for degree up to 10
degreerange = 10
degrees = range(1, degreerange + 1)
MSE_train = []
MSE_test = []

for deg in degrees:
    X = rt.create_X_polynomial(x, y, deg)
    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.25)
    beta = ols(X_train, z_train)
    ztilde_train = X_train @ beta
    ztilde_test = X_test @ beta
    
    MSE_train.append(rt.MSE(z_train, ztilde_train))
    MSE_test.append(rt.MSE(z_test, ztilde_test))

# Plotting MSE and R2 for all polynomial orders between 2 and 5
plt.plot(degrees, MSE_train, label="Train data MSE")
plt.plot(degrees, MSE_test, label="Test data MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("Mean Square Error")
plt.title("Train and test MSE as a function of model complexity")
plt.legend()
plt.show()
plt.clf()
print("Generated train v test MSE plot")

# Bootstrap for bias-variance tradeoff analysis
degrees = list(range(1, degreerange + 1))
biases, variances, errors = [], [], []
for deg in degrees:
    bias, variance, error = rt.bootstrap(x, y, z, deg, ols, B)
    biases.append(bias)
    variances.append(variance)
    errors.append(error)

plt.plot(degrees, biases, "--o" , label="Bias^2")
plt.plot(degrees, variances, "--o" ,label="Variance")
plt.plot(degrees, errors, "--o", label="Error")
plt.title("Bias-variance tradeoff for OLS")
plt.xlabel("Polynomial order")
plt.ylabel("Error scores")
plt.legend()
plt.show()
plt.clf()