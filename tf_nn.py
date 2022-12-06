
# This file corresponds to task c). It is based on M. Hjorth-Jensen's code,
# though heavily modified to use tensorflow and keras.
# STATUS: Not working. Central problem is that the tensorflow model
# does not seem to be able to differentiate the loss function. As far as we
# can tell, it doesn't have a set of variables to differentiate with respect to.
# These should be the weights and biases of the neural network, but calling
# tf.compat.v1.trainable_variables() returns an empty list.

# The following error is thrown when trying to differentiate the loss function:
# No gradients provided for any variable: ['dense/kernel:0', 'dense/bias:0', ...]

# We used a trick to allow for computation in g_trial, by feeding input as the
# true values. Anyone who would like to do this an alternative, I shall pray for you.



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend

import autograd.numpy as np
from autograd import jacobian,hessian,grad
import autograd.numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def u(x):
    """The initial condition"""
    return np.sin(np.pi*x)

## Define the trial solution and the right side of the ODE:
def g_trial(point, val):
    x, t = point
    return (1-t)*u(x) + x*(1-x)*t*val

# The right side of the ODE:
def f(x, t):
    return 0.0

#set up the neural network using tensorflow and keras
def predict_tf(x, t, model):
    """Predict the solution using a given model at the given points x and t"""
    prediction = model.predict(np.array([[x, t]]))
    return prediction

class Loss_tf(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):

        rhs = 0
        # Assign all the first elements of y_true to x_, and all the second elements
        # to t_.
        x_ = y_true[:,0]
        t_ = y_true[:,1]
        val = y_pred
    
        

        # Another issue we encountered was that val appears to be a keras tensor,
        # which is not compatible with autograd. We tried, and failed, to convert
        # it to a numpy array.

        point = np.array([x_[0],t_[0]])

        # Compute jacobian of g_trial with respect to x, t and val:
        g_t_jacobian_func = jacobian(g_trial)
        # Compute the heissian of g_trial with respect to x, t and val:
        g_t_hessian_func = hessian(g_trial)

        g_t_jacobian = g_t_jacobian_func(point, val)
        g_t_hessian = g_t_hessian_func(point, val)  

        
        s = []
        """
        for i, v in enumerate(val.flatten()):

            point = np.array([x_[i],t_[i]])

            # Compute jacobian of g_trial with respect to x, t and val:
            g_t_jacobian_func = jacobian(g_trial)
            # Compute the heissian of g_trial with respect to x, t and val:
            g_t_hessian_func = hessian(g_trial)

            g_t_jacobian = g_t_jacobian_func(point, v)
            g_t_hessian = g_t_hessian_func(point, v)            
                    
        
            g_t_dt = g_t_jacobian[0]
            g_t_d2x = g_t_hessian[1][1]

            y = g_t_dt - g_t_d2x 

            s.append(y)
        """


        y_pred = s
        y_true = rhs

        print(type(y_pred))
        print(type(y_true))

        quit()

        return backend.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)





def loss_tf(y_true, y_pred):

    # A cheeky trick is used here; whilst y_true should really be the right hand
    # side of the ODE, which is just zero, we instead feed it with the values of
    # x and t, which we then use in the g_trial function. As we know that y_true
    # is always zero, we can just set rhs = 0.
    
    rhs = 0
    x_ = y_true.numpy()[:,0]
    t_ = y_true.numpy()[:,1]
    val = y_pred.numpy()


    s = []
    
    for i, v in enumerate(val.flatten()):

        point = np.array([x_[i],t_[i]])

        # Compute jacobian of g_trial with respect to x, t and val:
        g_t_jacobian_func = jacobian(g_trial)
        # Compute the heissian of g_trial with respect to x, t and val:
        g_t_hessian_func = hessian(g_trial)

        g_t_jacobian = g_t_jacobian_func(point, v)
        g_t_hessian = g_t_hessian_func(point, v)            
                
    
        g_t_dt = g_t_jacobian[0]
        g_t_d2x = g_t_hessian[1][1]

        y = g_t_dt - g_t_d2x 

        s += (y - rhs)**2

    
    return s/len(val.flatten())



def solve_pde_deep_neural_network_tf(x, t, layers,
                                    activation="relu",
                                    output_activation="linear"):

    # Zip the meshgrid arrays x and t into one array of (x,t) points
    input = np.array([[x_, t_] for x_ in x for t_ in t])

    

    # Neural network is constructed using method add() from the Sequential class
    model = Sequential()

    # Input layer:
    model.add(Input(shape=(layers[0],)))

    # Hidden layers:
    for layer in layers[1:-1]:
        model.add(Dense(layer, activation=activation))

    # Output layer:
    model.add(Dense(layers[-1], activation=output_activation))
    
    # Model is compiled. Adam was chosen for no particular reason; loss function
    # is a custom function defined above. No idea what metrics is. Run_eagerly
    # is set to True, which we believed would help us convert the keras tensor
    # to a numpy array, but god is not so forgiving.
    model.compile(optimizer="adam", loss=Loss_tf(),  metrics=["mae"], run_eagerly=True)

    # model.summary() prints a summary of the model. This is useful for debugging.

    # Feeding the model with input twice is not a mistake; see loss_tf
    model.fit(input, input, epochs=10, verbose=1)

    return model



## For comparison, define the analytical solution, taken from the lecture notes:
def g_analytic(point):
    """Analytical solution to the heat equation"""
    x,t = point
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


if __name__ == '__main__':
    ### Use the neural network:
    npr.seed(15)

    ## Decide the vales of arguments to the function to solve
    Nx = 10; Nt = 10
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    input = np.array([[x_, t_] for x_ in x for t_ in t])
    
    ## Set up the parameters for the network
    layers = [2, 3, 1]
    num_iter = 250

    ## Solve the PDE. g_dnn_ag is the solution predicted by the final neural network.
    g_dnn_ag = np.zeros((Nx, Nt))
    model = solve_pde_deep_neural_network_tf(x, t, layers)


    # model.summary()
    # print("prediction of model: ", model.predict(np.array([[0.5, 0.5]])))
    
    for i,x_ in enumerate(x):
        for j, t_ in enumerate(t):
            g_dnn_ag[i,j] = g_trial(x_, t_, predict_tf(x_, t_, model))

    # Plotting code from the lecture notes:
    T,X = np.meshgrid(t,x)
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Solution from the deep neural network w/')
    s = ax.plot_surface(T,X,g_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$')
    plt.show()

    # plot g_trial with y_pred = 0 - usefull for debugging; checking if val does not simply output 0
    """
    g_dnn_ag = np.zeros((Nx, Nt))
    for i,x_ in enumerate(x):
        for j, t_ in enumerate(t):
            g_dnn_ag[i,j] = g_trial(x_, t_, 0)
    
    T,X = np.meshgrid(t,x)
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Solution from the deep neural network w/')
    s = ax.plot_surface(T,X,g_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$')
    plt.show()
    """
