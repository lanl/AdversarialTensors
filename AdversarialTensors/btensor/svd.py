import warnings
import torch

def svd_flip(U, V, u_based_decision=True):
    """
    Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    This function is borrowed from scikit-learn/utils/extmath.py

    Parameters
    ----------
    U : ndarray
        u and v are the output of SVD
    V : ndarray
        u and v are the output of SVD
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    batch_dim = U.shape[0]
    if u_based_decision:
        # columns of U, rows of V
        max_abs_cols = torch.argmax(torch.abs(U), axis=1)
        selected_max_el = torch.gather(U, 1, max_abs_cols.unsqueeze(1))
        signs = torch.sign(selected_max_el)
        U = U * signs
        if V.shape[1] > U.shape[2]:
            signs = torch.concat((signs, torch.ones(batch_dim,1, V.shape[1] - U.shape[2], device=U.device)), dim=2)
        V = V * signs[:,:,:V.shape[1]].transpose(1,2)
    else:
        # rows of V, columns of U
        max_abs_cols = torch.argmax(torch.abs(V), axis=2)
        selected_max_el = torch.gather(V, 2, max_abs_cols.unsqueeze(2))
        signs = torch.sign(selected_max_el)
        V = V * signs
        if U.shape[2] > V.shape[1]:
            signs = torch.concat((signs, torch.ones(batch_dim,U.shape[2] - V.shape[1], 1, device=U.device)), dim=1)

        U = U * signs[:,:U.shape[2],:]

    return U, V

def svd_checks(matrix, n_eigenvecs=None):
    """Runs common checks to all of the SVD methods.

    Parameters
    ----------
    matrix : 3D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    n_eigenvecs : int
        the number of eigenvectors to solve for
    min_dim : int
        the minimum dimension of matrix
    max_dim : int
        the maximum dimension of matrix
    """
    # Check that matrix is... a matrix!
    if matrix.ndim != 3:
        raise ValueError(f"batch of matricies is expected. matrix.ndim is {torch.ndim(matrix)} != 3")

    batch_dim, dim_1, dim_2 = matrix.shape
    min_dim, max_dim = min(dim_1, dim_2), max(dim_1, dim_2)

    if n_eigenvecs is None:
        n_eigenvecs = max_dim

    if n_eigenvecs > max_dim:
        warnings.warn(
            f"Trying to compute SVD with n_eigenvecs={n_eigenvecs}, which is larger "
            f"than max(matrix.shape)={max_dim}. Setting n_eigenvecs to {max_dim}."
        )
        n_eigenvecs = max_dim

    return n_eigenvecs, min_dim, max_dim


def truncated_svd(matrix, n_eigenvecs=None, **kwargs):
    """Computes a truncated SVD on `matrix` using the backends's standard SVD

    Parameters
    ----------
    matrix : 2D-array
    n_eigenvecs : int, optional, default is None
        if specified, number of eigen[vectors-values] to return

    Returns
    -------
    U : 2D-array
        of shape (matrix.shape[0], n_eigenvecs)
        contains the right singular vectors
    S : 1D-array
        of shape (n_eigenvecs, )
        contains the singular values of `matrix`
    V : 2D-array
        of shape (n_eigenvecs, matrix.shape[1])
        contains the left singular vectors
    """
    n_eigenvecs, min_dim, _ = svd_checks(matrix, n_eigenvecs=n_eigenvecs)
    full_matrices = True if n_eigenvecs > min_dim else False
    some = not full_matrices
    U, S, V = torch.svd(matrix, some=some)
    V = V.transpose(-1,-2)
    return U[:, :, :n_eigenvecs], S[:, :n_eigenvecs], V[:, :n_eigenvecs, :]

def randomized_svd(
    matrix,
    n_eigenvecs=None,
    n_oversamples=5,
    n_iter=2,
    random_state=0,
    **kwargs,
):
    """Computes a truncated randomized SVD.

    If `n_eigenvecs` is specified, sparse eigendecomposition is used on
    either matrix.dot(matrix.T) or matrix.T.dot(matrix).

    Parameters
    ----------
    matrix : tensor
        A 2D tensor.
    n_eigenvecs : int, optional, default is None
        If specified, number of eigen[vectors-values] to return.
    n_oversamples: int, optional, default = 5
        rank overestimation value for finiding subspace with better allignment
    n_iter: int, optional, default = 2
        number of power iterations for the `randomized_range_finder` subroutine
    random_state: {None, int, np.random.RandomState}
    **kwargs : optional
        kwargs are used to absorb the difference of parameters among the other SVD functions

    Returns
    -------
    U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
        Contains the right singular vectors
    S : 1-D tensor, shape (n_eigenvecs, )
        Contains the singular values of `matrix`
    V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
        Contains the left singular vectors

    Notes
    -----
    This function is implemented based on Algorith 5.1 in `Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions`
    - Halko et al (2009)
    """
    n_eigenvecs, min_dim, max_dim = svd_checks(matrix, n_eigenvecs=n_eigenvecs)

    batch_dim, dim_1, dim_2 = matrix.shape
    n_dims = min(n_eigenvecs + n_oversamples, max_dim)

    if (
        dim_1 > dim_2
        and n_eigenvecs > min(min_dim, n_dims)
        or dim_1 < dim_2
        and n_eigenvecs < min(min_dim, n_dims)
    ):
        # transpose matrix to keep the reduced matrix shape minimal
        matrix_T = torch.transpose(matrix, 1, 2)
        Q = randomized_range_finder(
            matrix_T, n_dims=n_dims, n_iter=n_iter, random_state=random_state
        )
        Q_H = torch.transpose(Q, 1, 2)
        matrix_reduced = torch.transpose(torch.matmul(Q_H, matrix_T))
        U, S, V = truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)
        V = torch.matmul(V, torch.transpose(Q, 1, 2))
    else:
        Q = randomized_range_finder(
            matrix, n_dims=n_dims, n_iter=n_iter, random_state=random_state
        )
        Q_H = torch.transpose(Q, 1, 2)
        matrix_reduced = torch.matmul(Q_H, matrix)
        U, S, V = truncated_svd(matrix_reduced, n_eigenvecs=n_eigenvecs)
        U = torch.matmul(Q, U)

    return U, S, V

def randomized_range_finder(A, n_dims, n_iter=2, random_state=0):
    """Computes an orthonormal matrix (Q) whose range approximates the range of A,  i.e., Q Q^H A ≈ A

    Parameters
    ----------
    A : 3D-array
    n_dims : int, dimension of the returned subspace
    n_iter : int, number of power iterations to conduct (default = 2)
    random_state: {None, int, np.random.RandomState}

    Returns
    -------
    Q : 3D-array
        of shape (batch, A.shape[0], min(n_dims, A.shape[0], A.shape[1]))

    Notes
    -----
    This function is implemented based on Algorith 4.4 in `Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions`
    - Halko et al (2009)
    """
    torch.manual_seed(random_state)
    batch, dim_1, dim_2 = A.shape
    Q = torch.normal(mean=0, std=1, size=(batch, dim_2, n_dims)).to(A.device)
    Q, _ = torch.qr(torch.matmul(A, Q))

    # Perform power iterations when spectrum decays slowly
    A_H = torch.transpose(A, 1, 2)
    for i in range(n_iter):
        Q, _ = torch.qr(torch.matmul(A_H, Q))
        Q, _ = torch.qr(torch.matmul(A, Q))

    return Q

SVD_FUNS = ["truncated_svd", "randomized_svd"]


def svd_interface(
    matrix,
    method="truncated_svd",
    n_eigenvecs=None,
    flip_sign=True,
    u_based_flip_sign=True,
    non_negative=None,
    mask=None,
    n_iter_mask_imputation=5,
    **kwargs,
):
    """Dispatching function to various SVD algorithms, alongside additional
    properties such as resolving sign invariance, imputation, and non-negativity.

    Parameters
    ----------
    matrix : tensor
        A 2D tensor.
    method : str, default is 'truncated_svd'
        Function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS or a callable.
    n_eigenvecs : int, optional, default is None
        If specified, number of eigen[vectors-values] to return.
    flip_sign : bool, optional, default is True
        Whether to resolve the sign indeterminacy of SVD.
    u_based_flip_sign : bool, optional, default is True
        Whether the sign indeterminacy should be resolved using U (vs. V).
    non_negative : bool, optional, default is False
        Whether to make the SVD results non-negative.
    nn_type : str, default is 'nndsvd'
        Algorithm to use for converting U to be non-negative.
    mask : tensor, default is None.
        Array of booleans with the same shape as ``matrix``. Should be 0 where
        the values are missing and 1 everywhere else. None if nothing is missing.
    n_iter_mask_imputation : int, default is 5
        Number of repetitions to apply in missing value imputation.
    **kwargs : optional
        Arguments passed along to individual SVD algorithms.

    Returns
    -------
    U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
        Contains the right singular vectors of `matrix`
    S : 1-D tensor, shape (n_eigenvecs, )
        Contains the singular values of `matrix`
    V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
        Contains the left singular vectors of `matrix`
    """

    if method == "truncated_svd":
        svd_fun = truncated_svd
    elif method == "randomized_svd":
        svd_fun = randomized_svd
    elif callable(method):
        svd_fun = method
    else:
        raise ValueError(
            f"Got svd={method}. However, the possible choices are {SVD_FUNS} or to pass a callable."
        )

    U, S, V = svd_fun(matrix, n_eigenvecs=n_eigenvecs, **kwargs)

    if flip_sign:
        U, V = svd_flip(U, V, u_based_decision=u_based_flip_sign)

    return U, S, V
