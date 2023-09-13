import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from AdversarialTensors.TensorDecomposition import  *
import time
verbose = True
data_modes = ['batch']#,'single']
data = torch.rand(1024,3,32,32)


for data_mode in data_modes:
    print('Working for data mode::',data_mode)
    params = {'init': 'svd', 'tol': 1e-3, 'factors': None, 'max_iter': 1000, 'svd': 'truncated_svd'
              }

    methods = ['tucker']
    if data_mode == 'single':
        ranks = [3, 5, 5]
    else:
        ranks = [32,3,5,5]
    for method in methods:
        t1 = time.time()
        model = TensorDecomposition(method=method, params=params, verbose=verbose,data_mode=data_mode)
        _, err = model(data,ranks )
        t2 = time.time()
        print('Took time=',t2-t1)
        print('Done testing for ', method)
