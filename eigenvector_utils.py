import numpy as np

    
def is_linearly_dependent(v, vs, tol=1e-2):
    """Check if a vector is linearly dependent on a set of vectors.
    
    Parameters:
        v (np.ndarray): the vector
        vs (list): the list of vectors
        
    Returns:
        bool: True if the vector is linearly dependent, False otherwise"""
    v = v / np.linalg.norm(v)
    for vi in vs:
        v -= v @ vi * vi
    return np.linalg.norm(v) < tol


def compute_eigenvalue(A, x):
    """Compute the eigenvalue of A given the eigenvector x.
    
    Parameters:
        A (np.ndarray): the matrix
        x (np.ndarray): the eigenvector
        
    Returns:
        float: the eigenvalue"""
    return x @ A @ x / (x @ x)

def is_eigenvector(A, v, tol=1e-2):
    """Check if a vector is an eigenvector of A.
    
    Parameters:
        A (np.ndarray): the matrix
        v (np.ndarray): the vector
        
    Returns:
        bool: True if the vector is an eigenvector, False otherwise"""
    return np.linalg.norm(A @ v - compute_eigenvalue(A, v) * v) < tol
