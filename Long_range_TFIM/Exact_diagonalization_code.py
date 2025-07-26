import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def sigma_x(i, L):
    """Construct sigma_x operator at site i."""
    eye = np.eye(2)
    sx = np.array([[0, 1], [1, 0]])
    op = 1
    for j in range(L):
        op = np.kron(op, sx if j == i else eye)
    return csr_matrix(op)

def sigma_z(i, L):
    """Construct sigma_z operator at site i."""
    eye = np.eye(2)
    sz = np.array([[1, 0], [0, -1]])
    op = 1
    for j in range(L):
        op = np.kron(op, sz if j == i else eye)
    return csr_matrix(op)

def build_tfim_hamiltonian(L, J=1.0, h=1.0, alpha=1.0):
    """Constructs the 1D long-range TFIM Hamiltonian."""
    H = csr_matrix((2**L, 2**L), dtype=np.float64)
    
    # Long-range ZZ interactions
    for i in range(L):
        for j in range(i+1, L):
            Jij = J / abs(i - j)**alpha
            H -= Jij * (sigma_z(i, L).dot(sigma_z(j, L)))
    
    # Transverse field X
    for i in range(L):
        H -= h * sigma_x(i, L)
    
    return H

L = 10            # Number of spins
J = 1.0          # Coupling constant
h = 1.0          # Transverse field
alpha = 0.1     # Power-law decay

H = build_tfim_hamiltonian(L, J, h, alpha)

# Compute ground state energy
eigs, vecs = eigsh(H, k=1, which='SA')  
print("Ground state energy:", eigs[0])