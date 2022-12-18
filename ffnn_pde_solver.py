import tensorflow as tf
import numpy as np

class EigenLoss(tf.keras.losses.Loss):
    def __init__(self, model):
        self.model = model
        super().__init__()
    #TODO

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
    xt = np.random.rand(1000, 2)
    solver = PDESolver([10, 10, 10], problem="diffusion")
    solver.fit(xt)
    print(solver.get_cost(xt))
