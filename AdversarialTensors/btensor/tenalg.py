from tensorly import unfold, partial_unfold, fold, partial_fold
import tensorly as tl
import torch
tl.set_backend('pytorch')

def batch_mode_dot(X, v, mode, transpose=False):
    """
    Perform batched mode-dot product along a specific mode.

    Parameters:
    - X: The input tensor
    - v: The matrix to be multiplied
    - mode: The mode along which to multiply
    - transpose: Whether to transpose the matrix v before multiplication

    Returns:
    - The result of the mode-dot operation
    """
    new_shape = list(X.shape)
    if transpose:
        v = torch.transpose(v, 1, 2)
    # update the output shape in case the v's rank is reduced
    # +1 becuase there is a batch dimension
    new_shape[mode+1] = v.shape[1]
    res = torch.matmul(v, partial_unfold(X, mode))
    return partial_fold(res, mode, shape=new_shape)

def batch_multi_mode_dot(X, v, modes=None, transpose=False, skip=None):
    """
    Perform batched multi-mode dot product.

    Parameters:
    - X: The input tensor
    - v: The list of matrices to be multiplied
    - modes: The modes along which to multiply each matrix
    - transpose: Whether to transpose the matrices before multiplication
    - skip: If specified, the mode to skip during multiplication

    Returns:
    - The result of the multi-mode dot operation
    """
    if modes is None:
        modes = range(len(v))
    res = X
    factors_modes = sorted(zip(v, modes), key=lambda x: x[1])
    for i, (matrix_or_vec, mode) in enumerate(factors_modes):
        if (skip is not None) and (i == skip):
            continue

        res = batch_mode_dot(res, matrix_or_vec, mode, transpose=transpose)
    return res

def batch_ttm(X, v, mode, transpose=False):
    """
    Perform batched tensor-times-matrix (TTM) multiplication along given mode(s).

    Parameters:
    - X: The input tensor
    - v: The matrix (or list of matrices) to be multiplied
    - mode: The mode(s) along which to multiply
    - transpose: Whether to transpose the matrix before multiplication

    Returns:
    - The result of the TTM operation
    """
    if  isinstance(mode,int):
        return batch_mode_dot(X, v, mode=mode, transpose=transpose)
    else:
        return batch_multi_mode_dot(X, v, modes=mode, skip=None, transpose=transpose)

def batched_regress_core(data, factors):
    """
    Perform batched regression on the core tensor.

    Parameters:
    - data: The input tensor
    - factors: The factor matrices

    Returns:
    - The core tensor after regression
    """
    return batch_ttm(data, factors, range(len(factors)), transpose=True)

def batched_reconstruct(data, factors):
    """
    Perform batched reconstruction of tensor from its factors.

    Parameters:
    - data: The input tensor
    - factors: The factor matrices

    Returns:
    - The reconstructed tensor
    """
    return batch_ttm(data, factors, range(len(factors)), transpose=False)
    
def batched_naive_rank_estimator(eigsumthresh, eigvals):
    """
    Batched estimation of tensor rank based on a given eigenvalue sum threshold.

    Parameters:
    - eigsumthresh: The eigenvalue sum threshold
    - eigvals: The eigenvalues

    Returns:
    - Estimated ranks for each sample in the batch
    """
    # calculate cumulative sum of singular values (decreasing order) for each sample
    # in the batch
    eigsums = torch.cumsum(eigvals.flip(dims=(1,)), 1)
    # find the first index above the threshold
    ignored_ranks = torch.argmax((eigsums > eigsumthresh).type(torch.int32),dim=1)
    full_rank = eigvals.shape[1]
    ranks = full_rank - ignored_ranks
    return ranks
