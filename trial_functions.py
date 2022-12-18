import numpy as np

# For the diffusion equation
def u0(x):
    """Initial condition for the diffusion equation.
    
    Parameters:
        x (np.ndarray): the x values
    
    Returns:
        np.ndarray: the initial condition"""
    return np.sin(np.pi * x)


def u_trial(x, t, N):
    """Trial solution for the diffusion equation.
    
    Parameters:
        x (np.ndarray): the x values
        t (np.ndarray): the t values
    
    Returns:
        np.ndarray: the trial solution"""
    return (1-t) * u0(x) + t * x * (1-x) * N


def u_analytical(x, t):
    """Return the analytical solution of the heat equation with initial condition u0
    
    Parameters:
        x (np.ndarray): the x values
        t (np.ndarray): the t values
        
    Returns:
        np.ndarray: the analytical solution"""
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

# For the eigenvalue problem 

def generate_symmetric(N=6, seed=89, old_version=False):
    """Generate a N x N symmetric matrix."""
    np.random.seed(seed)
    if old_version:
        A = np.random.rand(N, N)
    else:
        A = np.random.randn(N, N)
    return (A + A.T) / 2

def x_trial_normalized(t, x0, N):
    """Trial solution for the eigenvalue problem.
    
    Parameters:
        t (np.ndarray): the t values
        x0 (np.ndarray): the initial condition
        N (np.ndarray): the neural network output
        
    Returns:
        np.ndarray: the normalized trial solution"""
    print("x0", x0.shape)
    print("N", N.shape)
    print("t", t.shape)
    x0 = x0.reshape(1, -1)
    t = t.reshape(-1, 1)
    x = np.exp(-t) * x0 + (1 - np.exp(-t)) * N
    return x / np.linalg.norm(x)