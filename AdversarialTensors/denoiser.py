#@author: Manish Bhattarai, Mehmet Cagri

import numpy as np
import torch.nn
import torch
import tensorly as tl
tl.set_backend('pytorch')
from .TensorDecomposition import *
from .patcher import *

class Denoiser(torch.nn.Module):
    def __init__(self, method='tucker', device='cuda',tensor_params={'factors':None,'svd':'truncated_svd', 'init':'svd','tol':1e-5,'max_iter':1000},
                 verbose=True,patch_params={'patch_size':8, 'stride':4,'channels':3},data_mode='single',ranks=None):
        """
        Initializes the Denoiser module.

        Parameters:
        -----------
        method: str, optional (default='tucker')
            The tensor decomposition method to use. Valid options are 'tucker' and 'parafac'.

        device: int, optional (default=0)
            The GPU device to use for computations.

        tensor_params: dict, optional (default={'factors': None, 'init': 'svd', 'tol': 1e-5, 'max_iter': 1000})
            Dictionary containing parameters for the tensor decomposition method.

        verbose: bool, optional (default=True)
            Whether to print verbose output during computation.

        patch_params: dict, optional (default={'patch_size': 8, 'stride': 4, 'channels': 3})
            Dictionary containing parameters for patch transformation of data.

        data_mode: str, optional (default='single')
            The data mode, whether to use single or double precision arithmetic.

        ranks: list, optional (default=None)
            Rank for decomposition
        """

        super(Denoiser, self).__init__()
        self.method = method
        self.tensor_params = tensor_params
        self.verbose = verbose
        self.data_mode =data_mode
        self.patch_params = patch_params
        self.device = device
        self.tensor_model = TensorDecomposition(method=self.method,params=self.tensor_params,verbose=self.verbose,data_mode=self.data_mode)
        self.patcher = patch_transform(**self.patch_params)
        self.ranks = ranks

    def forward(self,X,ranks=None, recon_err=False):
        """
        Computes the forward pass of the denoiser.

        Parameters:
        -----------
        X: tensor, shape (batch_size, channels, height, width)
            The input tensor.

        ranks: list of ints, length 5
            The ranks to use for the tensor decomposition. The ranks should be for dimensions N x P x C x W x H.

        Returns:
        --------
        X_recon: tensor, shape (batch_size, channels, height, width)
            The output tensor, denoised by the tensor decomposition.

        err: float
            The error between the input and output tensors.
        """
        if self.ranks is None:
            self.ranks = ranks
        #Here ranks should be for dims NXPXCXWXH i.e 5 valued multirank vector
        X = self.patcher.fit(X,mode='patch')
        X_recon,err = self.tensor_model(X,self.ranks)
        X_recon = self.patcher.fit(X_recon,mode='merge')
        if recon_err:
            return X_recon, err
        else:
            return X_recon


