import tensorflow as tf
import numpy as np


class EigenLoss(tf.keras.losses.Loss):
    """This is a loss function for the differential equation described in a
    paper by Yi and Fu (2004). https://doi.org/10.1016/S0898-1221(04)90110-1.
    """
    def __init__(self, model, x0, A):
        """Initialize the loss function.

        Parameters:
            model (keras model): the model to be trained"""
        self.model = model
        self.x0 = x0
        self.A = A
        super().__init__()


    def f_tf(self, x):
        """The function f(x) as described in the above paper.
        
        Parameters:
            x (tensor): the x values
        
        Returns:
            tensor: the function f(x)"""
        A = self.A
        N = A.shape[0]
        x2 = tf.einsum("ij, ij -> i", x, x)
        t1 = tf.einsum("i, jk -> ijk", x2, A)
        xAx = tf.einsum("ij, ij -> i", x@A, x)

        t2 = tf.einsum("i, jk -> ijk", (1 - xAx), tf.eye(N))
        sol = tf.einsum("ijk, ik -> ij", t1 + t2, x)
        return sol

    def x_trial_tf(self, t):
        """Trial value for vector valued function x(t).
        
        Parameters:
            x (tensor): the x values
            t (tensor): the t values
        
        Returns:
            tensor: the right hand side"""
        #x = (1 - t) * self.x0 + t * self.model(t)
        x = tf.exp(-t) * self.x0 + (1 - tf.exp(-t)) * self.model(t)
        return x

    def rhs(self, t):
        """Right hand side of the differential equation.
        
        Parameters:
            t (tensor): the t values
        
        Returns:
            tensor: the right hand side"""
        x_trial = self.x_trial_tf(t)
        return self.f_tf(x_trial) - x_trial


    def __call__(self, t, y_pred, sample_weight=None):
        """Compute the loss for the diffusion equation.
        
        Parameters:
            t (tensor): time values
            y_pred (tensor): the function N(t) given by the FFNN
            sample_weight (tensor): To match the keras loss function signature.
                Not in use.
        Returns:
            tensor: the loss (MSE between lhs and rhs of the diffusion equation)"""
        x_trial = self.x_trial_tf(t)
        
        with tf.GradientTape() as grad:
            grad.watch(t)
            x_trial = self.x_trial_tf(t)
        dxdt = grad.gradient(x_trial, t)

        diff = dxdt - self.rhs(t) # lhs - rhs
        return tf.reduce_mean(tf.square(diff))


class DiffusionLoss(tf.keras.losses.Loss):
    """This is a loss function for the diffusion equation. 
    It subclasses from keras loss functions."""

    def __init__(self, model):
        """Initialize the loss function.

        Parameters:
            model (keras model): the model to be trained"""
        self.model = model
        super().__init__()
    
    def u0_tf(self, x):
        """Initial condition for the diffusion equation.

        Parameters:
            x (tensor): the x values
            
        Returns:
            tensor: the initial condition"""
        return tf.sin(np.pi * x)

    def u_trial_tf(self, x, t):
        """Trial solution for the diffusion equation.
        
        Parameters:
            x (tensor): the x values
            t (tensor): the t values
        
        Returns:
            tensor: the trial solution"""
        xt = tf.stack([x, t], axis=1)
        return (1-t) * self.u0_tf(x) + t * x * (1-x) * self.model(xt)
    
    def __call__(self, xt, y_pred, sample_weight=None):
        """Compute the loss for the diffusion equation.
        
        Parameters:
            xt (tensor): the x and t values
            y_pred (tensor): the predicted values. not in use.
            sample_weight (tensor): To match the keras loss function signature.
                Not in use.
        Returns:
            tensor: the loss (MSE between lhs and rhs of the diffusion equation)"""
        x, t = xt[:, 0], xt[:, 1] 
        with tf.GradientTape() as grad1:
            grad1.watch([x, t])
            with tf.GradientTape(persistent=True) as grad2:
                grad2.watch([x, t])

                g = self.u_trial_tf(x, t)

            dgdx = grad2.gradient(g, x)
            dgdt = grad2.gradient(g, t)
        d2gdx2 = grad1.gradient(dgdx, x)

        return tf.reduce_mean(tf.square(d2gdx2 - dgdt))

class PDESolver:

    def __init__(
        self, 
        hidden_layers, 
        learning_rate=0.001,
        regularization=0.,
        activation_function="tanh",
        problem="diffusion",
        A=None,
        x0=None
        ):
        """Initialize the PDESolver class.
        
        Parameters:
            hidden_layers (list): list of the number of nodes in each hidden layer
            learning_rate (float): learning rate for the optimizer
            regularization (float): regularization parameter hidden layer weights
            activation_function (str): activation function to use in the hidden layers.
                default is 'tanh', but can be any of the keras activation functions.
            problem (str): the problem to solve. Either 'diffusion' or 'eigen'
            A (np.ndarray): the matrix A in the eigenvalue problem. Only used if
                problem is 'eigen'
            x0 (np.ndarray): the initial condition in the eigenvalue problem. Only
                used if problem is 'eigen'
            """
        regularizer = tf.keras.regularizers.l2(regularization)
        if problem == "diffusion":
            input_size = 2
            output_size = 1
        elif problem == "eigen":
            assert A is not None
            assert x0 is not None
            input_size = 1
            output_size = A.shape[0]

        layers = [tf.keras.layers.Dense(
            hidden_layers[0], 
            activation=activation_function,
            kernel_regularizer=regularizer,
            input_shape=(input_size,)
            )]
        for i in range(1, len(hidden_layers)):
            layers.append(tf.keras.layers.Dense(
                hidden_layers[i], 
                activation=activation_function,
                kernel_regularizer=regularizer
                ))
        layers.append(tf.keras.layers.Dense(output_size))

        self.model = tf.keras.models.Sequential(layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if problem == "diffusion":
            self.loss_fn = DiffusionLoss(self.model)
        elif problem == "eigen":
            A = tf.convert_to_tensor(A, dtype=float)
            x0 = tf.convert_to_tensor(x0, dtype=float)
            self.loss_fn = EigenLoss(self.model, x0, A)
        else:
            raise ValueError("Problem must be either 'diffusion' or 'eigen'")

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss_fn
        )


    def fit(self, x, y=None, **kwargs):
        """Fits the model to the data

        Parameters:
            x (tensor): design matrix
            y (tensor): target values. Not in use.
            **kwargs: keyword arguments to pass to the keras fit function.
                See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit"""
        x = tf.convert_to_tensor(x, dtype=float)
        x = tf.expand_dims(x, axis=-1)
        self.model.fit(x, x, **kwargs)


    def __call__(self, x, **kwargs):
        """Gives model predictions
        
        Parameters:
            x (tensor): design matrix
            **kwargs: keyword arguments to pass to the keras predict function.
                See https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
            
        Returns:
            tensor: model predictions"""
        return self.model(x, **kwargs).numpy()
    
    
    def get_cost(self, x):
        """Compute the cost function for the model.
        
        Parameters:
            x (np.ndarray): design matrix
        
        Returns:
            tensor: the cost function"""
        x = tf.convert_to_tensor(x, dtype=float)
        x = tf.expand_dims(x, axis=-1)
        return self.loss_fn(x, x).numpy()
    

if __name__ == "__main__":
    from trial_functions import x_trial_normalized, generate_symmetric
    N = 3
    A = generate_symmetric(N)
    x0 = np.random.rand(N)

    solver = PDESolver([10, 10, 10], problem="eigen", A=A, x0=x0)
    t = np.linspace(0, 30, 100)
    solver.fit(t)
    
    N = solver(t)
    x_trial = x_trial_normalized(t, x0, N)
    import matplotlib.pyplot as plt

    plt.plot(t, x_trial)
    plt.show()
