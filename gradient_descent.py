#%%
import math

import numpy as np
from abc import ABC, abstractmethod


class GradientDescent:

    def __init__(
        self,
        batch_size = None,
        momentum_param = 0,
        mode = "normal",
        store_extra=False
    ):
        """Initialize a gradient descent object.
        
        Parameters
        ----------
        batch_size : int, optional
            Size of each batch, by default None (equivalent to batch_size = size of dataset)
        momentum_param : float, optional
            Momentum parameter, by default 0 (no momentum)
        mode : str, optional
            Mode of gradient descent, by default "normal".
                "normal" : normal gradient descent
                "adagrad" : Adagrad
                "rmsprop" : RMSProp
                "adam" : Adam
        store_extra : bool, optional
            Whether to store cost function and weight values for each epoch, by default False
        """
        # Batch size of None corresponds to gradient descent
        self.batch_size = batch_size

        # Momentum discount factor
        self.momentum_param = momentum_param
        self.momentum = 0

        if mode not in ["normal", "adagrad", "rmsprop", "adam"]:
            raise ValueError("Mode must be one of 'normal', 'adagrad', 'rmsprop', 'adam'")
        self.mode = mode
        self.store_extra = store_extra
        self.has_trained = False


    def train(self, X, w, y, model, learning_rate, n_epochs):
        """Train a model using gradient descent.
        
        Parameters
        ----------
        X : np.array
            Design matrix
        w : np.array
            Initial weight vector
        y : np.array
            Target vector
        model : Model
            Model to train
        learning_rate : float
            Learning rate
        n_epochs : int
            Number of epochs to train for
        """
        if self.has_trained:
            raise Exception("This gradient descent object has already been used to train a model. Please create a new object.")
        
        self.X = X
        self.w = w.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        self.model = model
        self.eta = learning_rate

        if self.batch_size is None:
            batch_size = len(y)
        else:
            batch_size = self.batch_size

        if self.store_extra:
            # Store cost function and weight values for each epoch
            # Store cost and weight values before training
            self.costs = [model.cost(self.w, X, self.y)]
            self.weights = [w.flatten()]
            
        for epoch in range(1, n_epochs + 1):
            # Shuffle data
            if self.batch_size:
                idx = np.random.permutation(len(self.y))
                self.X = self.X[idx]
                self.y = self.y[idx]
            # For each batchm, update weights
            for batch in range(math.ceil(len(y) / batch_size)):
                self.X_batch = self.X[batch * batch_size: (batch + 1) * batch_size]
                self.y_batch = self.y[batch * batch_size: (batch + 1) * batch_size]
                self.w = self.update()
            # Store cost and weight values after each epoch
            if self.store_extra:
                self.costs.append(model.cost(self.w, X, y))
                self.weights.append(self.w.flatten())
        
        return self.w

    def delta_w(self, X, w, y, model, eta):
        return self.momentum * self.momentum_param - eta * model.gradient(w, X, y)

    def delta_w_adam(self):
        beta1 = 0.9
        beta2 = 0.999
        g = self.model.gradient(self.w, self.X_batch, self.y_batch)

        if not hasattr(self, "m"):
            self.m = np.zeros((len(self.w), 1))
            self.s = np.zeros((len(self.w), 1))
            self.t = 0

        self.t += 1
        
        self.m = beta1 * self.m + (1 - beta1) * g
        self.s = beta2 * self.s + (1 - beta2) * (g**2)


        m_hat = self.m / (1 - beta1**self.t)
        s_hat = self.s / (1 - beta2**self.t)

        eps = 1e-8
        pre_momentum = self.eta * m_hat / (np.sqrt(s_hat) + eps)

        return self.momentum * self.momentum_param - pre_momentum

    def adagrad(self):
        if not hasattr(self, "G"):
            self.G = np.zeros((len(self.w), 1))
        epsi = 1e-8
        self.G += self.model.gradient(self.w, self.X_batch, self.y_batch)**2
        return self.eta / np.sqrt(self.G + epsi)
    
    def rmsprop(self):
        beta = 0.9

        g = self.model.gradient(self.w, self.X_batch, self.y_batch)

        if not hasattr(self, "s"):
            self.s = np.zeros((len(self.w), 1))
        self.s = beta * self.s + (1 - beta) * g**2
        eps = 1e-8
        return self.eta / np.sqrt(self.s + eps)
    
    def update(self):
        X = self.X_batch
        w = self.w
        y = self.y_batch
        model = self.model
        eta = self.eta

        if self.mode == "adagrad":
            eta_star = self.adagrad()
        elif self.mode == "rmsprop":
            eta_star = self.rmsprop()
        else:
            eta_star = eta

        if self.mode == "adam":
            delta_w = self.delta_w_adam()
        else:
            delta_w = self.delta_w(X, w, y, model, eta_star)
        
        self.momentum = delta_w
        return w + delta_w        

