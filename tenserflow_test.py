import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

#Want to solve pratial differential equation using tensorflow

import autograd.numpy as np
from autograd import jacobian,hessian,grad
import autograd.numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


#set up the neural network using tensorflow and keras
def predict_tf(x, t, model):
    prediction = model(np.array([x, t])).numpy()
    print("prediction: ", prediction)
    return prediction

def solve_pde_deep_neural_network_tf(x, t, layers, output,
                                    activation="relu",
                                    output_activation="linear"):
    input = np.array([x, t])

    #set up the neural network using tensorflow and keras
    model = Sequential()

    #input layer:
    model.add(tf.keras.Input(shape=(layers[0],)))

    #hidden layers:
    for layer in layers[1:-1]:
        model.add(Dense(layer, activation=activation))

    #output layer:
    model.add(Dense(layers[-1], activation=output_activation))
    
    #compile the model
    model.compile(optimizer="adam", loss=cost_func_tf, metrics=["mae"])

    #train the model
    model.fit(input, output, epochs=1000, verbose=0)

    return model


## Define the trial solution and cost function
def u(x):
    return np.sin(np.pi*x)

def g_trial(x, t, model):
    return (1-t)*u(x) + x*(1-x)*t*predict_tf(x, t, model)

# The right side of the ODE:
def f(x, t):
    return 0.

# The cost function:
def get_y_pred(x, t, model):

    g_t_jacobian_func = jacobian(g_trial)
    g_t_hessian_func = hessian(g_trial)

    for x_ in x:
        for t_ in t:
            point = np.array([x_,t_])

            g_t = g_trial(x, t, model)

            g_t_jacobian = g_t_jacobian_func(x, t, model)
            g_t_hessian = g_t_hessian_func(x, t, model)

            g_t_dt = g_t_jacobian[1]
            g_t_d2x = g_t_hessian[0][0]

    return g_t_dt - g_t_d2x 

def cost_func_tf(y_true, y_pred):
    return (y_pred - y_true)**2

## For comparison, define the analytical solution
def g_analytic(point):
    x,t = point
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


if __name__ == '__main__':
    