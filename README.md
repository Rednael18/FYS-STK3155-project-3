# FYS-STK3155 Project 3
## Authors:
- Carl Fredrik Nordbø Knutsen
- Benedict Leander Sk ̊alevik Parton
- Halvor Tyseng

## Description
Project description can be found at: https://compphysics.github.io/MachineLearning/doc/Projects/2022/Project3/pdf/Project3.pdf.

The project is in essence about solving differential equations using two methods, namely forward Euler and feedforward neural network-based approximations. We will examine the 1D diffusion equation and an eigenvalue dynamical system, described in https://doi.org/10.1016/S0898-1221(04)90110-1.

## Structure
The solvers are implemented in the following files:
- ffnn_pde_solver.py contains loss functions and a class for wrapping tensor flow FFNNs (it interfaces nicely with numpy!)
- euler.py contains the forward Euler method for the diffusion equation
- eigenvalue_solver.py contains code for finding eigenvalues (minimization problem)
  - feedforward_nn.py is used for this (our own implementation of a feedforward neural network)
  - gradient_descent.py is used for this (our own implementation of gradient descent)

## Analysis
The analysis can be found in the notebooks in the notebooks folder.

## Usage
The differential equation solver can be used as follows for the eigenvalue dynamical system:
```python
from ffnn_pde_solver import PDESolver
from trial_functions import x_trial_normalized, generate_symmetric

N = 3
A = generate_symmetric(N)
x0 = np.random.rand(N)

solver = PDESolver([10, 10, 10], problem="eigen", A=A, x0=x0)
t = np.linspace(0, 30, 100)
solver.fit(t)

nn_output = solver(t)
x_trial = x_trial_normalized(t, x0, nn_output)
```
similar usage for the diffusion equation.