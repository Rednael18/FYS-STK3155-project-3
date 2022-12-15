import numpy as np
import matplotlib.pyplot as plt
import data_generation as dg
import regression_tools as rt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer as wbc


x, y, z, _ = dg.generate_data_Franke(1000, 0.1, seed=1)
X = np.array([[x[i], y[i]] for i in range(len(x))])
X_train, X_test, z_train, z_test = train_test_split(X, y, test_size=0.2)
z_train = np.around(z_train*50)
z_test = np.around(z_test*50)



"""
# Load data
data = wbc()
X = data.data
y = data.target

# Split data into training and test sets
X_TR, X_test, z_TR, z_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, z_train, z_val = train_test_split(X_TR, z_TR, test_size=0.2)
n = X_train.shape[1]
"""


max_depth_range = 15
bias = []
variance = []  # What do?

for i in range(1, max_depth_range):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf.fit(X_train, z_train)

    pred = clf.predict(X_test)
    bias.append(1 - rt.accuracy(z_test, pred))
    tree.plot_tree(clf, filled=True)
    #plt.show()
    plt.clf()

plt.plot(np.linspace(1, max_depth_range, max_depth_range-1), bias)
plt.show()


