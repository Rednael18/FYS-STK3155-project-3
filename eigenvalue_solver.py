import numpy as np

from feedforward_nn import NeuralNetwork
from gradient_descent import GradientDescent


def generate_symmetric(N=6, seed=89):
    """Generate a N x N symmetric matrix."""
    np.random.seed(seed)
    A = np.random.rand(N, N)
    return (A + A.T) / 2


def compute_eigenvalues(A, eigenvectors):
    """Compute the eigenvalues of A.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to compute the eigenvalues of.
    eigenvectors : np.ndarray
        The eigenvectors of A.
    
    Returns
    -------
    eigenvalues : np.ndarray
        The eigenvalues of A."""
    if A.shape != eigenvectors.shape:
        raise ValueError("A and eigenvectors must have the same shape")
    
    N = A.shape[0]
    eigenvalues = []
    for i in range(N):
        vector = eigenvectors[:, i].reshape(N, 1)
        value = vector.T @ A @ vector / (vector.T @ vector)
        eigenvalues.append(value[0, 0])
    return np.array(eigenvalues)
    

def compute_eigenvalues_eigenvectors(
    A,
    epochs=20000,
    learning_rate=0.005,
    weight_scaling=100
    ):
    """Computes all eigenvalues and eigenvectors of A.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to compute the eigenvalues and eigenvectors of.
    epochs : int
        The number of epochs to train each neural network for.
    learning_rate : float
        The learning rate for the gradient descent.
    
    Returns
    -------
    eigenvalues : np.ndarray
        The eigenvalues of A.
    eigenvectors : np.ndarray
        The eigenvectors of A."""
    found_vectors = []
    N = A.shape[0]

    cnt = 0
    while len(found_vectors) < N:
        cnt += 1
        if cnt > 30:
            print("Failed to find all eigenvectors: Too many iterations")
            break

        x0 = np.random.rand(1, N)
        for eigenvector in found_vectors:
            x0 = x0 - (x0 @ eigenvector.T) / np.linalg.norm(eigenvector) * eigenvector
        x0 = x0 / np.linalg.norm(x0)

        # Run the computation
        nn = NeuralNetwork([N, 20, 20, N], activation="relu", cost_function="eigen", A=A)
        wb = nn.wb() / weight_scaling
        gd = GradientDescent(mode="adam", momentum_param=0.5, store_extra=True)   
        wb = gd.train(x0, wb, x0, nn, learning_rate, epochs)
        eigenvector_candidate = x0 + nn.predict(wb, x0)
        eigenvector_candidate = eigenvector_candidate / np.linalg.norm(eigenvector_candidate)

        # Check if the eigenvector is valid
        if gd.costs[-1] > 0.00001:
            print(f"Failed to find eigenvector: Cost too high (iteration{cnt})")
            continue
        
        x = eigenvector_candidate
        for eigenvector in found_vectors:
            x -= (x @ eigenvector.T) / np.linalg.norm(eigenvector) * eigenvector
        if np.linalg.norm(x) < 0.99:
            print(f"Failed to find eigenvector: Linearly dependent (iteration {cnt})")
            continue
        
        found_vectors.append(eigenvector_candidate)
        print(f"Found eigenvector {len(found_vectors)} (iteration {cnt})")
    found_vectors = [vec.T for vec in found_vectors]
    for i in range(N):
        if found_vectors[i][0] < 0:
            found_vectors[i] = -found_vectors[i]
    found_vectors.sort(key=lambda x: x[0])

    # Turn the eigenvectors into a matrix
    eigenvectors = np.concatenate(found_vectors, axis=1)
    eigenvalues = compute_eigenvalues(A, eigenvectors)
    return eigenvalues, eigenvectors


def main():
    """Example showcasing the eigenvalue solver and comparing with numpy's
    eigenvalue solver."""
    N = 6
    A = generate_symmetric(N)
    nnval, nnvec = compute_eigenvalues_eigenvectors(A)
    print(nnval)
    print(nnvec)
    print()

    val, vec = np.linalg.eig(A)
    for i in range(N):
        if vec[0, i] < 0:
            vec[:, i] = -vec[:, i]

    idx = sorted([(vec[0, i], i) for i in range(N)])
    idx = [i[1] for i in idx]
    val = val[idx]
    vec = vec[:, idx]
    print(val)
    print(vec)


if __name__ == "__main__":
    main()
