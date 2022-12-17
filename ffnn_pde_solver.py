import tensorflow as tf
import numpy as np

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

    def g_trial_tf(self, x, t):
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

                g = self.g_trial_tf(x, t)
                # print("g", g)

            dgdx = grad2.gradient(g, x)
            dgdt = grad2.gradient(g, t)
        d2gdx2 = grad1.gradient(dgdx, x)

        return tf.reduce_mean(tf.square(d2gdx2 - dgdt))

class PDESolver:

    def __init__(
        self, 
        hidden_layers, 
        activation_function="relu",
        learning_rate=0.001,
        regularization=0.
        ):
        """Initialize the PDESolver class.
        
        Parameters:
            hidden_layers (list): list of the number of nodes in each hidden layer
            activation_function (str): activation function to use in the hidden layers
            optimizer (str): optimizer to use for training. Default is adam."""
        regularizer = tf.keras.regularizers.l2(regularization)

        layers = [tf.keras.layers.Dense(
            hidden_layers[0], 
            activation=activation_function,
            kernel_regularizer=regularizer,
            input_shape=(2,)
            )]
        for i in range(1, len(hidden_layers)):
            layers.append(tf.keras.layers.Dense(
                hidden_layers[i], 
                activation=activation_function,
                kernel_regularizer=regularizer
                ))
        layers.append(tf.keras.layers.Dense(1))

        self.model = tf.keras.models.Sequential(layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = DiffusionLoss(self.model)
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
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        x = tf.expand_dims(x, axis=-1)
        self.model.fit(x, x, **kwargs)


    def __call__(self, x, **kwargs):
        """Gives model predictions
        
        Parameters:
            x (tensor): design matrix"""
        return self.model(x, **kwargs).numpy()