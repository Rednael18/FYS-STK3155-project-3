
# This file consists of task a) and b); solving the heat equation with the forward Euler method


import numpy as np
import matplotlib.pyplot as plt


L = 1  # Length of the rod


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
        ax.title.set_text('Temperature over time and space')
        plt.show()
    
    return u

def analytical_solution(x, t):
    """Return the analytical solution of the heat equation with initial condition u0"""
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


compute_temperature_gradient(0.00001, 0.01, 1, plot=True)

# Plot analytical solution
x = np.linspace(0, L, 100)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, analytical_solution(X, T), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.title.set_text('Analytical solution')
plt.show()

# Plot the error
x = np.linspace(0, L, 101)
t = np.linspace(0, 1, 100000)
X, T = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, analytical_solution(X, T) - compute_temperature_gradient(0.00001, 0.01, 1, plot=False), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.title.set_text('Forward Euler error')
plt.show()

    
