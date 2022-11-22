import numpy as np
import matplotlib.pyplot as plt

L = 1


def u0(x):
    """Return the initial x-value at time t=0."""
    if type(x) == int:
        if x < 0 or x > L:
            raise Warning("x is not in [0, L]")
    elif type(x) == np.ndarray:
        if np.any(x < 0) or np.any(x > L):
            raise Warning("x is not in [0, L]")
    return np.sin(np.pi * x)


def forward_euler(u0, dt, dx, T, stoptime=True):
    """Return the solution of the heat equation with initial condition u0
    using the forward Euler method, with a centered difference in space, time step dt 
    and space step dx. The solution is computed for T seconds."""
    # Number of time steps. If stoptime, T is the final time, otherwise T is the number of steps.
    if stoptime:
        N = int(T / dt)
    else:
        N = T
    # Number of space steps
    M = int(L / dx)
    # Initialize the solution array
    u = np.zeros((N + 1, M + 1))
    # Set the initial condition
    u[0, :] = u0(np.linspace(0, L, M + 1))
    # Set the boundary conditions
    u[:, 0] = 0
    u[:, -1] = 0
    # Compute the solution
    for n in range(N):
        u[n + 1, 1:-1] = u[n, 1:-1] + dt / dx**2 * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
    return u


def compute_temperature_gradient(dt, dx, T, stoptime=True, plot=True):
    """Return the temperature gradient at the center of the rod for the solution of the heat equation"""
    # Compute the solution
    u = forward_euler(u0, dt, dx, T)

    if plot:
        # Plot the solution
        x = np.linspace(0, L, int(L / dx) + 1)
        t = np.linspace(0, T, int(T / dt) + 1)
        X, T = np.meshgrid(x, t)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, T, u, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.show()


    
