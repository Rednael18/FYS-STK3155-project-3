import numpy as np
    
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
    """Return the analytical solution of the heat equation with initial condition u0"""
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)