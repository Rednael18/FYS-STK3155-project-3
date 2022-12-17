
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


def solve_diffusion_equation_FE(dt, dx, T, plot=False):
    """Solves the 1D diffusion equation using the forward Euler method, 
    with time step dt, space step dx, and u0 as initial condition.
    """
    # Compute the solution
    u = forward_euler(u0, dt, dx, T)

    if plot:
        # Plot the solution
        x = np.linspace(0, L, int(L / dx) + 1)
        t = np.linspace(0, T, int(T / dt) + 1)
        X, T = np.meshgrid(x, t)
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.plot_surface(T, X, u, cmap='viridis')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u')
        ax.title.set_text('Forward Euler Approximation of the Diffusion Equation')
        plt.show()
    
    return u

def analytical_solution(x, t):
    """Return the analytical solution of the heat equation with initial condition u0"""
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)


def main():
    solve_diffusion_equation_FE(0.00001, 0.01, 1, plot=True)

    # Plot analytical solution
    x = np.linspace(0, L, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, T, analytical_solution(X, T), cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.title.set_text('Analytical solution')
    plt.show()

    # Plot the absolute error
    x = np.linspace(0, L, 101)
    t = np.linspace(0, 1, 100000)
    X, T = np.meshgrid(x, t)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    absolute_error = np.absolute(
        analytical_solution(X, T) 
        - solve_diffusion_equation_FE(0.00001, 0.01, 1)
        )
    ax.plot_surface(X, T, absolute_error, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.title.set_text('Forward Euler absolute error')
    plt.show()
    print("Maximum absolute error: ", np.max(absolute_error))

        
if __name__ == "__main__":
    main()