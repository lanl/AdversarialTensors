import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from AdversarialTensors.TensorDecomposition import *

verbose = True
data_modes=['batch']

for data_mode in data_modes:
    params = {'init': 'svd', 'tol': 1e-3, 'factors': None, 'max_iter': 1000, 'svd': 'truncated_svd','data_mode':data_mode}

    methods = ['tucker', 'ntucker', 'cpd', 'ncpd', 'tt', 'nnsvd', 'svd', 'nmf']

    core = torch.rand(4, 4, 4)
    factors = [torch.rand(100, 4) for _ in range(3)]
    from tensorly import tucker_to_tensor

    data = tucker_to_tensor((core, factors))
    for method in ['tucker', 'ntucker']:
        model = TensorDecomposition(method=method, params=params, verbose=verbose)
        _, err = model(data, ranks=[4, 5, 6])
        print('Done testing for ', method)

    for method in ['cpd', 'ncpd', 'tt']:
        model = TensorDecomposition(method=method, params=params, verbose=verbose)
        if method=='ttcross':
            ranks =  [1,4,4,1]
        else:
            ranks = 4
        _, err = model(data, ranks=ranks)
        print('Done testing for ', method)

    params = {'init': 'nnsvd', 'tol': 1e-7, 'factors': None, 'max_iter': 1000, 'svd': 'truncated_svd'}
    for method in ['svd', 'nnsvd', 'nmf']:
        model = TensorDecomposition(method=method, params=params, verbose=verbose)
        _, err = model(data.reshape(-1, np.product(data.shape[1:])), ranks=4)
        print('Done testing for ', method)
