import numpy as np

def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def generate_data_Franke(N, sigma2, seed):
    """Generates N uniformly distributed points on [0,1] x [0,1]
    applies the Franke Function to the points and adds noise

    Parameters
    ----------
    N: int
        Number of points
    sigma2: float
        Variance of error term 
    seed: int
        seed

    Returns
    -------
    x: np.ndarray
        N uniformly distributed x-coordinates
    y: np.ndarray
        N uniformly distributed y-coordinates
    z: np.ndarray
        Franke Function applied to x and y plus noise
    z_true:
        Franke Function applied to x and y, without noise
    """
    np.random.seed(seed)

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    z_true = franke_function(x, y)
    epsilon = np.random.normal(0, sigma2, (N)) # generates n samples epsilon ~ N(0, sigma2)
    z = z_true + epsilon

    return x, y, z, z_true


def linear_function(x, y):
    cx = 1
    cy = 2
    return cx * x + cy * y


def generate_data_linear(N, sigma2, seed):
    """Generates N uniformly distributed points on [0,1] x [0,1]
    applies a linear function to the points and adds noise
    """
    np.random.seed(seed)

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    z_true = linear_function(x, y)
    epsilon = np.random.normal(0, sigma2, (N)) # generates n samples epsilon ~ N(0, sigma2)
    z = z_true + epsilon

    X = np.vstack([x, y]).T

    return X, z, z_true


def generate_data_binary(N, seed=987, add_col_ones=False):
    """Generates N variable z with values 0 or 1,
    that is 1 if x > 0.5 and y > 0.5, else 0.

    Parameters
    ----------
    N: int
        Number of points
    seed: int
        seed
    """
    np.random.seed(seed)

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    z = np.zeros(N)
    for i in range(N):
        if x[i] > 0.5 and y[i] > 0.5:
            z[i] = 1
        else:
            z[i] = 0

    if add_col_ones:
        X = np.vstack([np.ones(N), x, y]).T
    else:
        X = np.vstack([x, y]).T
    z = z.reshape(-1, 1)

    return X, z


def generate_and_data():
    """Generates n points of the AND function"""
    x = [1, 1, 0, 0]
    y = np.array([1, 0, 1, 0])
    z = np.array([1, 0, 0, 0])
    X = np.vstack([x, y]).T
    return X, z.reshape(-1, 1)


def generate_xor_data():
    """Generates n points of the XOR function"""
    x = np.array([1, 1, 0, 0])
    y = np.array([1, 0, 1, 0])
    z = np.array([0, 1, 1, 0])
    X = np.vstack([x, y]).T
    return X, z.reshape(-1, 1)


def main():
    N = 10
    X, y = generate_data_binary(N, add_col_ones=True)
    print(X)


if __name__ == "__main__":
    main()