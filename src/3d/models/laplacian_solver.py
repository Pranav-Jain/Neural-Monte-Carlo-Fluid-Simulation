import numpy as np
from scipy.sparse.linalg import factorized
from scipy import sparse


def factorized_laplacian_solver(N):
    return factorized(build_laplacian_matrix(N, N))


def build_laplacian_matrix(M: int, N: int):
    """build laplacian matrix with zero boundary condition (eliminate degree of freedom)

    Args:
        M (int): number of rows
        N (int): number of cols

    Returns:
        np.array: laplacian matrix 
    """
    main_diag = np.full(M * N, -4)
    main_diag[[0, N - 1, -N, -1]] = -2
    main_diag[[*range(1, N - 1), *range(-N + 1, -1), 
               *range(N, (M - 2) * N + 1, N), *range(2 * N - 1, (M - 1) * N, N)]] = -3
    side_diag = np.ones(M * N - 1)
    side_diag[[*range(N - 1, M * N - 1, N)]] = 0
    data = [np.ones(M * N - N), side_diag, main_diag, side_diag, np.ones(M * N - N)]
    offsets = [-N, -1, 0, 1, N]
    mat = sparse.diags(data, offsets)
    return mat