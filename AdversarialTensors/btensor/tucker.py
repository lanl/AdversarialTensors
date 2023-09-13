import numpy as np
import torch.nn
from tensorly import unfold, partial_unfold, fold, partial_fold
import tensorly as tl
from .svd import svd_interface
from .tenalg import batch_multi_mode_dot
import warnings
tl.set_backend('pytorch')

def initialize_tucker(
    tensor,
    rank,
    modes,
    random_state,
    init="svd",
    svd="truncated_svd",
    non_negative=False,
):
    r"""
    Initializes core and factors used in the Tucker decomposition.
    The type of initialization is set using the \texttt{init} parameter.
    If \texttt{init} is set to 'random', factor matrices are initialized using \texttt{random_state}.
    If \texttt{init} is set to 'svd', the \( m \)th factor matrix is initialized using the
    \texttt{rank} left singular vectors of the \( m \)th unfolding of the input tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to be decomposed.
    rank : int
        Number of components.
    modes : list of int
        List of modes for Tucker decomposition.
    random_state : int
        Random seed for initialization.
    init : {'svd', 'random'}, optional
        Type of initialization ('svd' or 'random').
    svd : str, default is 'truncated\_svd'
        Function to use for SVD, acceptable values in tensorly.SVD\_FUNS.
    non\_negative : bool, default is False
        If True, non-negative factors are returned. Has no effect for now.

    Returns
    -------
    tuple
        A tuple containing:
        - core (torch.Tensor): Initialized core tensor.
        - factors (list): List of initialized factor matrices.
    """

    # Initialisation
    if init == "svd":
        factors = []
        for index, mode in enumerate(modes):
            U, _, _ = svd_interface(
                partial_unfold(tensor, mode),
                n_eigenvecs=rank[index],
                method=svd,
                non_negative=non_negative,
                mask=None,
                n_iter_mask_imputation=0,
                random_state=random_state,
            )

            factors.append(U)
        # The initial core approximation is needed here for the masking step
        core = batch_multi_mode_dot(tensor, factors, modes=modes, transpose=True)

    elif init == "random":
        batch_size = tensor.shape[0]
        torch.manual_seed(random_state)
        core_dims = [batch_size,] + list(rank)
        core = torch.rand(core_dims) + 0.01
        core = core.to(tensor.device)
        factors = []
        for s in zip(tensor.shape[1:], rank):
            dims = [batch_size,] + list(s)
            factor = torch.rand(dims).to(tensor.device)
            factors.append(factor)

    else:
        (core, factors) = init

    return core, factors

def tucker(
    tensor,
    rank,
    fixed_factors=None,
    n_iter_max=100,
    init="svd",
    return_errors=False,
    svd="truncated_svd",
    tol=10e-5,
    random_state=0,
    mask=None,
    verbose=False,
):
    """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition:
        ``tensor = [| core; factors[0], ...factors[-1] |]`` [1]_

    Parameters
    ----------
    tensor : torch.tensor
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim - 1)``
        if int, the same rank is used for all modes
    fixed_factors : int list or None, default is None
        if not None, list of modes for which to keep the factors fixed.
        Only valid if a Tucker tensor is provided as init.
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, optional
    return_errors : boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD,
        acceptable values in tensorly.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True).
    verbose : int, optional
        level of verbosity

    Returns
    -------
    core : torch.tensor of size `ranks`
            core tensor of the Tucker decomposition
    factors : torch.tensor list
            list of factors of the Tucker decomposition.
            Its ``i``-th element is of shape ``(tensor.shape[i], ranks[i])``

    References
    ----------
    .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    if fixed_factors:
        try:
            (core, factors) = init
        except:
            raise ValueError(
                f'Got fixed_factor={fixed_factors} but no appropriate Tucker tensor was passed for "init".'
            )

        fixed_factors = sorted(fixed_factors)
        modes_fixed, factors_fixed = zip(
            *[(i, f) for (i, f) in enumerate(factors) if i in fixed_factors]
        )
        core = batch_multi_mode_dot(core, factors_fixed, modes=modes_fixed)
        modes, factors = zip(
            *[(i, f) for (i, f) in enumerate(factors) if i not in fixed_factors]
        )
        init = (core, list(factors))

        (core, new_factors), rec_errors, avg_rec_errors = partial_tucker(
            tensor,
            rank=rank,
            modes=modes,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            random_state=random_state,
            mask=mask,
            verbose=verbose,
        )

        factors = list(new_factors)
        for i, e in enumerate(fixed_factors):
            factors.insert(e, factors_fixed[i])
        core = batch_multi_mode_dot(core, factors_fixed, modes=modes_fixed, transpose=True)

        return (core, factors)

    else:
        modes = list(range(tensor.ndim - 1))
        #TODO: rank validation is removed
        # rank = validate_tucker_rank(tl.shape(tensor), rank=rank)

        (core, factors), rec_errors, avg_rec_errors = partial_tucker(
            tensor,
            rank=rank,
            modes=modes,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            random_state=random_state,
            mask=mask,
            verbose=verbose,
        )
        tensor = (core, factors)
        if return_errors:
            return tensor, avg_rec_errors
        else:
            return tensor


def partial_tucker(
    tensor,
    rank,
    modes=None,
    n_iter_max=100,
    init="svd",
    tol=10e-5,
    svd="truncated_svd",
    random_state=0,
    verbose=False,
    mask=None,
    svd_mask_repeats=5,
):
    """Partial tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes `tensor` into a Tucker decomposition exclusively along the provided modes.

    Parameters
    ----------
    tensor : ndarray
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    modes : None, int list
            list of the modes on which to perform the decomposition
    n_iter_max : int
                 maximum number of iteration
    init : {'svd', 'random'}, or TuckerTensor optional
        if a TuckerTensor is provided, this is used for initialization
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD,
        acceptable values in tensorly.tenalg.svd.SVD_FUNS
    tol : float, optional
          tolerance: the algorithm stops when the variation in
          the reconstruction error is less than the tolerance
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        level of verbosity
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True).

    Returns
    -------
    core : ndarray
            core tensor of the Tucker decomposition
    factors : ndarray list
            list of factors of the Tucker decomposition.
            with ``core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes``

    """
    if modes is None:
        modes = list(range(tensor.ndim) - 1)

    if rank is None:
        message = "No value given for 'rank'. The decomposition will preserve the original size."
        warnings.warn(message, Warning)
        rank = [tensor.shape[mode] for mode in modes]
    elif isinstance(rank, int):
        message = f"Given only one int for 'rank' instead of a list of {len(modes)} modes. Using this rank for all modes."
        warnings.warn(message, Warning)
        rank = tuple(rank for _ in modes)
    else:
        rank = tuple(rank)

    # SVD init
    core, factors = initialize_tucker(
        tensor,
        rank,
        modes,
        init=init,
        svd=svd,
        random_state=random_state,
    )

    rec_errors = []
    avg_rec_errors = []
    dims = [d + 1 for d in range(tensor.ndim - 1)]
    norm_tensor = torch.norm(tensor,2,dim=dims)


    for iteration in range(n_iter_max):
        for index, mode in enumerate(modes):
            core_approximation = batch_multi_mode_dot(
                tensor, factors, modes=modes, skip=index, transpose=True
            )
            eigenvecs, _, _ = svd_interface(
                partial_unfold(core_approximation, mode),
                n_eigenvecs=rank[index],
                random_state=random_state,
            )
            factors[index] = eigenvecs

        core = batch_multi_mode_dot(tensor, factors, modes=modes, transpose=True)
        # The factors are orthonormal and therefore do not affect the reconstructed tensor's norm
        dims = [d + 1 for d in range(tensor.ndim - 1)]
        rec_error = torch.sqrt(torch.abs(norm_tensor**2 - torch.norm(core, 2, dim=dims) ** 2)) / norm_tensor
        avg_rec_errors.append(torch.mean(rec_error))
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print(
                    f"Avg. reconstruction error={avg_rec_errors[-1]}, variation={avg_rec_errors[-2] - avg_rec_errors[-1]}."
                )

            if tol and torch.all(torch.abs(rec_errors[-2] - rec_errors[-1]) < tol):
                if verbose:
                    print(f"converged in {iteration} iterations.")
                break

    return (core, factors), rec_errors, avg_rec_errors
