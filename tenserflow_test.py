import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

#Want to solve pratial differential equation using tensorflow

import autograd.numpy as np
from autograd import jacobian,hessian,grad
import autograd.numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def u(x):
    return np.sin(np.pi*x)

## Define the trial solution and the right side of the ODE:
def g_trial(x, t, y_pred):
    print("do we get here?")
    return (1-t)*u(x) + x*(1-x)*t*y_pred

# The right side of the ODE:
def f(x, t):
    return 0.0

#set up the neural network using tensorflow and keras
def predict_tf(x, t, model):
    prediction = model.predict(np.array([[x, t]]))
    print("prediction: ", prediction)
    return prediction

def loss_tf(y_true, y_pred):
    global x
    global t
    #convert y_pred from tensor to numpy array
    g_t_jacobian_func = jacobian(g_trial)
    g_t_hessian_func = hessian(g_trial)

    print("bars...")
    print(y_pred)
    print(y_pred.numpy())
    print(np.size(y_pred.numpy()))
    s = 0 
    for x_ in x:
        for t_ in t:
            g_t_jacobian = g_t_jacobian_func(x_, t_, y_pred)
            print("bars... one")
            print("g_t_jacobian: ", g_t_jacobian)
            g_t_hessian = g_t_hessian_func(x_, t_, y_pred)
            print("bars... two")
            
            
            print("g_t_hessian: ", g_t_hessian)

            g_t_dt = g_t_jacobian[1]
            g_t_d2x = g_t_hessian[0][0]

    y = g_t_dt - g_t_d2x 
    
    return np.mean(np.square(y - y_true))


def solve_pde_deep_neural_network_tf(x, t, layers,
                                    activation="relu",
                                    output_activation="linear"):

    input = np.array([[x_, t_] for x_ in x for t_ in t])
    

    #set up the neural network using tensorflow and keras
    global model
    model = Sequential()
    print('one')

    
    #input layer:
    model.add(Input(shape=(layers[0],)))

    #hidden layers:
    for layer in layers[1:-1]:
        model.add(Dense(layer, activation=activation))

    #output layer:
    model.add(Dense(layers[-1], activation=output_activation))
    
    #compile the model
    model.compile(optimizer="adam", loss=loss_tf,  metrics=["mae"], run_eagerly=True)

    true_value = np.zeros(len(x)*len(t))
    print(np.shape(true_value))
    model.summary()
    #train the model
    model.fit(input, true_value, epochs=10, verbose=1)
    print('three')
    return model



## For comparison, define the analytical solution
def g_analytic(point):
    x,t = point
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


if __name__ == '__main__':
    ### Use the neural network:
    npr.seed(15)

    ## Decide the vales of arguments to the function to solve
    Nx = 10; Nt = 10
    global x
    global t
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    input = np.array([[x_, t_] for x_ in x for t_ in t])
    
    ## Set up the parameters for the network
    layers = [2, 5, 5, 1]
    num_iter = 250

    ## Solve the PDE
    g_dnn_ag = np.zeros((Nx, Nt))
    model = solve_pde_deep_neural_network_tf(x, t, layers)
    model.summary()
    
    print("prediction of model: ", model.predict(np.array([[0.5, 0.5]])))
    
    for i,x_ in enumerate(x):
        for j, t_ in enumerate(t):
            g_dnn_ag[i,j] = g_trial(x_, t_, predict_tf(x_, t_, model))

    T,X = np.meshgrid(t,x)
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.set_title('Solution from the deep neural network w/')
    s = ax.plot_surface(T,X,g_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$')
    plt.show()

    # plot g_trial with y_pred = 0
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
