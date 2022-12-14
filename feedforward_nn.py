import autograd.numpy as np
from autograd import jacobian, hessian, grad
from autograd import elementwise_grad as egrad


class NeuralNetwork():

    def __init__(
        self, 
        layers, 
        activation="relu",
        output_activation="linear",
        cost_function = "mse",
        regularization=0,
        A=None
        ):
        """Initialize a neural network with the given layers and activation function.
        Parameters
        ----------
        layers: list
            List of integers, where each integer represents the number of nodes in a layer.
        activation: str
            Activation function to use for all layers except the input and final layers.
            Options are "sigmoid", "relu", "leaky_relu", and "linear".
        output_activation: str
            Activation function to use for the final layer.
            Options are "sigmoid", "relu", "leaky_relu", and "linear".
        cost_fn: str
            Cost function to use.
            Options are "mse" and "cross_entropy".
        regularization: float
            L2 regularization parameter.
        """
        self.layers = layers
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

        # Activation functions
        activation_dict = {
            "sigmoid": self._sigmoid,
            "relu": self._relu,
            "leaky_relu": self._leaky_relu,
            "linear": self._linear
        }

        # Set input layer activation function to be linear.
        self.input_activation_fn = activation_dict["linear"]

        # Set hidden layer activation function
        self.activation_fn = activation_dict[activation]

        # Set final layer activation function
        self.final_activation_fn = activation_dict[output_activation]

        # Set cost function
        if cost_function == "mse":
            self.cost_fn = self.mse
        elif cost_function == "diffusion":
            self.cost_fn = self.diffusion_cost
        elif cost_function == "eigen":
            self.cost_fn = self.eigen_cost
        else:
            raise ValueError("Invalid cost function.")

        self.regularization = regularization

        self.A = A
    
    def _initialize_weights(self):
        weights = []
        for i in range(len(self.layers) - 1):
            weights.append(np.random.normal(0, 1, (self.layers[i], self.layers[i+1])))
        return weights

    def _initialize_biases(self):
        biases = []
        for i in range(len(self.layers) - 1):
            #TODO: make zero?
            biases.append(np.zeros((1, self.layers[i+1])))
        return biases

    def _linear(self, z):
        return z

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _relu(self, z):
        return np.maximum(z, 0)

    def _leaky_relu(self, z):
        return np.where(z > 0, z, z * 0.01)

    def get_activation_fn(self, layer):
        if layer == 0:
            return self.input_activation_fn
        elif layer == len(self.layers) - 1:
            return self.final_activation_fn
        else:
            return self.activation_fn

    def diffusion_trial(self, xt, wb):
        x, t = xt
        u = lambda xx: np.sin(np.pi*xx)
        N = self.predict(wb, xt)
        return (1-t) * u(x) + t * x * (1-x) * N


    def diffusion_cost(self, wb):
        x, t = self.X[:, 0], self.X[:, 1]

        cost_sum = 0

        g_t_jacobian_func = jacobian(self.diffusion_trial)
        g_t_hessian_func = hessian(self.diffusion_trial)

        for x_, t_ in zip(x,t):
            point = np.array([x_,t_])
            g_t = self.diffusion_trial(point,wb)
            g_t_jacobian = g_t_jacobian_func(point,wb).flatten()
            g_t_hessian = g_t_hessian_func(point,wb)
            
            g_t_dt = g_t_jacobian[1]
            g_t_d2x = g_t_hessian[0,0,0,0]

            err_sqr = (g_t_dt - g_t_d2x)**2
            cost_sum += err_sqr

        return cost_sum /( np.size(x)*np.size(t) )

    def f_trial(self, x):
        A = self.A
        N = A.shape[0]
        return ((x.T @ x ) * A + (1 - x.T @ A @ x) * np.identity(N)) @ x

    def eigen_cost(self, output):
        x0 = self.X
        x = x0 + output
        x = (x / np.linalg.norm(x)).T
        f = self.f_trial(x)
        return self.mse(x, f)
    
    def mse(self, y_pred, y_true):
        """Mean squared error cost function."""
        return np.mean((y_pred - y_true)**2)

    def _forward_propagation(self):
        """Perform forward propagation to compute the activations and
        activation inputs."""
        W = self.weights
        b = self.biases
        
        Z = [] # activation inputs
        A = [] # activation values

        z = self.X
        for layer in range(len(self.layers)):
            activation = self.get_activation_fn(layer)
            a = activation(z)
            
            A.append(a)
            Z.append(z)

            if layer != len(self.layers) - 1:
                z = a @ W[layer] + b[layer]    
        
        return A, Z
        
    def concatenated_weights_and_biases(self):
        """reshapes the weights and biases into a single column vector."""
        wb = np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])
        return wb.reshape(-1, 1)

    def wb(self):
        return self.concatenated_weights_and_biases()

    def unflatten_weights_and_biases(self, wb):
        """Unflattens the weights and biases from a single vector.
        Parameters
        ----------
        w_and_b: np.ndarray
            Vector of weights and biases.
        Returns
        -------
        weights: list
            List of weight matrices.
        biases: list
            List of bias vectors.
        """
        wb = wb.flatten()
        weights = []
        biases = []
        start = 0
        for i in range(len(self.layers) - 1):
            end = start + self.layers[i] * self.layers[i+1]
            weights.append(wb[start:end].reshape(self.layers[i], self.layers[i+1]))
            start = end
        for i in range(len(self.layers) - 1):
            end = start + self.layers[i+1]
            biases.append(wb[start:end].reshape(1, self.layers[i+1]))
            start = end
        return weights, biases

    def cost(self, wb, X, y):
        weights, biases = self.unflatten_weights_and_biases(wb)
        y = y.reshape(-1, 1)

        self.X = X
        self.weights = weights
        self.biases = biases
        self.y = y

        activations, _ = self._forward_propagation()
        if self.cost_fn == self.diffusion_cost:
            cost = self.cost_fn(wb)
        elif self.cost_fn == self.eigen_cost:
            cost = self.cost_fn(activations[-1])
        else:
            cost = self.cost_fn(activations[-1], y)

        # Add L2 regularization
        if self.regularization > 0:
            reg = self.regularization * np.sum(np.concatenate([w.flatten() ** 2 for w in weights]))
            cost += reg

        return cost

    def gradient(self, wb, X, y):
        weights, biases = self.unflatten_weights_and_biases(wb)

        self.X = X
        self.weights = weights
        self.biases = biases
        self.y = y

        
        # Compute gradient using autograd
        gradient = egrad(self.cost)
        return gradient(wb, X, y)

    def predict(self, wb, X):
        weights, biases = self.unflatten_weights_and_biases(wb)

        self.X = X
        self.weights = weights
        self.biases = biases
        activations, _ = self._forward_propagation()
        return activations[-1]


def nn_example():
    # Example case
    nn = NeuralNetwork([2, 2, 1], cost_function="diffusion")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    wb = nn.wb()
    print("Initial weights and biases: ", wb.shape)
    print("Initial cost: ", nn.cost(wb, X, y))
    print("Initial gradient: ", nn.gradient(wb, X, y))

def pde_solver_example():
    nn = NeuralNetwork([2, 2, 1], cost_function="diffusion")
    x = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 10)
    X, T = np.meshgrid(x, t)
    # save shape of X and T
    xt_shape = X.shape

    X = X.flatten()
    T = T.flatten()
    X = np.array([X, T]).T
    y = np.zeros(X.shape[0])
    wb = nn.wb()
    from time import time
    start = time()
    print("Initial weights and biases: ", wb.shape)
    print("Initial cost: ", nn.cost(wb, X, y))
    print("Initial gradient: ", nn.gradient(wb, X, y))
    print("Time taken: ", time() - start)


    # Run gradient descent
    from gradient_descent import GradientDescent
    import matplotlib.pyplot as plt
    gd = GradientDescent(store_extra=True)
    wb = gd.train(X, wb, X[:, 0], nn, 0.1, 10)
    plt.plot(gd.costs)
    plt.show()

    # Compute solution at all points using diffusion_trial
    sol = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        sol[i] = nn.diffusion_trial(X[i], wb)
    
    # Reshape solution to original shape
    sol = sol.reshape(xt_shape)
    plt.imshow(sol)
    plt.show()


def generate_symmetric(N=6, seed=89):
    """Generate a symmetric matrix."""
    #np.random.seed(seed)
    A = np.random.rand(N, N)
    return (A + A.T) / 2

def eigvec_eigval_example():
    from gradient_descent import GradientDescent

    eigvec = np.array([-0.45195742 ,-0.63577615 ,-0.44818869 ,-0.43663496])   
    N = 4
    A = generate_symmetric(N, seed=280)
    nn = NeuralNetwork([N, 20, 20, N], activation="relu", cost_function="eigen", A=A)
    # print("eigencost", nn.eigen_cost(eigvec))
    #np.random.seed(9996)
    x = np.random.rand(1, N)
    # A = np.array([[1, 2, 3, 4],
    #              [2, 3, 4, 5],
    #              [3, 4, 5, 6],
    #              [4, 5, 6, 7]])
    print(A)
    print(x)
    print("Initial cost: ", nn.cost(nn.wb(), x, np.array([0])))
    print("Initial prediction: ", nn.predict(nn.wb(), x) )

    # perform gradient descent
    wb = nn.wb()
    gd = GradientDescent(store_extra=True)


    wb = gd.train(x, wb, x, nn, 0.01, 4000)
    import matplotlib.pyplot as plt
    plt.plot(gd.costs)
    plt.show()
    print("HOMEGROWN:")
    v_pred = x + nn.predict(wb, x)
    print(v_pred / np.linalg.norm(v_pred))

    #print(np.linalg.norm(A @ v_pred) / np.linalg.norm(v_pred))
    print("NUMPY:")
    val, vec = np.linalg.eig(A)
    print(val)
    print(vec)


def main():
    #pde_solver_example()
    eigvec_eigval_example()
if __name__ == "__main__":
    main()