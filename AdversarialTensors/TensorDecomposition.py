#@author: Manish Bhattarai, Mehmet Cagri
import numpy as np
import torch.nn
import torch
import tensorly as tl
tl.set_backend('pytorch')


class TensorDecomposition(torch.nn.Module):
    r"""
    A class to perform tensor decomposition using various methods such as CPD, Tucker, TT, etc.

    Parameters
    ----------
    method : str, optional
        The method to use for decomposition. Default is 'tucker'.
    params : dict, optional
        Dictionary of parameters for the decomposition method.
        Default is {'factors': None, 'init': 'svd', 'tol': 1e-5, 'n_iter_max': 1000}.
    verbose : bool, optional
        Whether or not to print status messages during decomposition. Default is True.
    data_mode : str, optional
        Whether the tensor is in batch mode ('batch') or single mode ('single'). Default is 'batch'.

    Attributes
    ----------
    method : str
        The decomposition method.
    params : dict
        Dictionary of decomposition method parameters.
    X : torch.Tensor
        The input tensor.
    ranks : tuple or int
        Desired ranks for decomposition.
    reconstruct : bool
        Whether to reconstruct tensor after decomposition.
    verbose : bool
        Verbose flag.
    data_mode : str
        Data mode ('batch' or 'single').
    """
    def __init__(self, method='tucker', params={'factors':None,'init':'svd','svd':'truncated_svd','tol':1e-5,'n_iter_max':1000},verbose=True,data_mode='batch'):
        """
         Initialize the TensorDecomposition class.

         Parameters
         ----------
         method : str, optional
             The method to use for decomposition. Default is 'tucker'.
         params : dict, optional
             Dictionary of parameters for the decomposition method.
             Default is {'factors': None, 'init': 'svd', 'tol': 1e-5, 'n_iter_max': 1000}.
         verbose : bool, optional
             Whether or not to print status messages during decomposition. Default is True.
         data_mode : str, optional
             Whether the tensor is in batch mode ('batch') or single mode ('single'). Default is 'batch'.
         """
        super(TensorDecomposition, self).__init__()
        self.method = method
        self.params = params
        self.X = None
        self.ranks = None
        self.reconstruct  = True
        self.verbose = verbose
        self.data_mode = data_mode


    def tucker(self):
        import time
        """
        Performs Tucker decomposition using tensorly library.

        Returns
        -------
        tuple
            A tuple containing:
            - core (torch.Tensor): the core tensor
            - factors (list): a list of factor matrices
            - X_recon (torch.Tensor, optional): the reconstructed tensor, if self.reconstruct=True
            - err (float, optional): the relative error between X and X_recon, if self.reconstruct=True
        """
        if self.data_mode=='batch':
            from tensorly.decomposition import tucker
            core,factors = tucker(self.X,fixed_factors=self.params['factors'],rank=self.ranks,verbose=self.verbose,init=self.params['init'],tol=self.params['tol'])
        elif self.data_mode == 'single':
            from .btensor.tucker import tucker as tucker_parallel
            from .btensor.tenalg import batched_reconstruct as tucker_to_tensor_parallel
            (core, factors) = tucker_parallel(self.X, fixed_factors=self.params['factors'], rank=self.ranks, verbose=self.verbose,
                                   init=self.params['init'], tol=self.params['tol'], return_errors=False, svd=self.params['svd'], random_state=int(time.time()))
        if self.reconstruct:
            if self.data_mode=='batch':
                from tensorly import tucker_to_tensor
                X_recon = tucker_to_tensor((core,factors))
            elif self.data_mode == 'single':
                from .btensor.tenalg import batched_reconstruct as tucker_to_tensor_parallel
                X_recon = tucker_to_tensor_parallel(core,factors)
            err = self.relative_error(self.X,X_recon)
            if self.verbose: print('Reconstruction Error::',err)
            return (core,factors),X_recon,err
        else:
            return (core,factors)

    def ntucker(self):
        r"""
        Performs Non-negative Tucker decomposition using the tensorly library.

        Returns
        -------
        tuple
            A tuple containing:
            - core (torch.Tensor): the core tensor
            - factors (list): a list of factor matrices
            - X_recon (torch.Tensor): the reconstructed tensor (if self.reconstruct=True)
            - err (float): the relative error between X and X_recon (if self.reconstruct=True)
        """
        from tensorly.decomposition import non_negative_tucker
        core,factors = non_negative_tucker(self.X,rank=self.ranks,verbose=self.verbose,init=self.params['init'],tol=self.params['tol'],n_iter_max=self.params['max_iter'])
        if self.reconstruct:
            from tensorly import tucker_to_tensor
            X_recon = tucker_to_tensor((core,factors))
            err = self.relative_error(self.X,X_recon)
            if self.verbose: print('Reconstruction Error::',err)
            return (core,factors),X_recon,err
        else:
            return (core,factors)

    def cpd(self):
        r"""
        Performs CPD decomposition using tensorly library.

        Returns
        -------
        tuple
            A tuple containing:
            - core (torch.Tensor): the core tensor
            - factors (list): a list of factor matrices
            - X_recon (torch.Tensor): the reconstructed tensor (if self.reconstruct=True)
            - err (float): the relative error between X and X_recon (if self.reconstruct=True)
        """
        from tensorly.decomposition import parafac
        core,factors = parafac(self.X,rank=self.ranks,verbose=self.verbose,init=self.params['init'],tol=self.params['tol'],n_iter_max=self.params['max_iter'])
        if self.reconstruct:
            from tensorly import cp_to_tensor
            X_recon = cp_to_tensor((core,factors))
            err =self.relative_error(self.X,X_recon)
            if self.verbose: print('Reconstruction Error::',err)
            return (core,factors),X_recon,err
        else:
            return (core,factors)

    def ncpd(self):
        r"""
        Performs Non-negative CPD decomposition using tensorly library.

        Returns
        -------
        tuple
            A tuple containing:
            - core (torch.Tensor): the core tensor
            - factors (list): a list of factor matrices
            - X_recon (torch.Tensor): the reconstructed tensor (if self.reconstruct=True)
            - err (float): the relative error between X and X_recon (if self.reconstruct=True)
        """
        from tensorly.decomposition import non_negative_parafac
        core,factors = non_negative_parafac(self.X,rank=self.ranks,verbose=self.verbose,init=self.params['init'],tol=self.params['tol'],n_iter_max=self.params['max_iter'])
        if self.reconstruct:
            from tensorly import cp_to_tensor
            X_recon = cp_to_tensor((core,factors))
            err = self.relative_error(self.X,X_recon)
            if self.verbose: print('Reconstruction Error::',err)
            return (core,factors),X_recon,err
        else:
            return (core,factors)

    def tt(self):
        r"""
        Performs Tensor-train decomposition using tensorly library.

        Returns
        -------
        tuple
            A tuple containing:
            - factors (list): a list of factor matrices
            - X_recon (torch.Tensor): the reconstructed tensor (if self.reconstruct=True)
            - err (float): the relative error between X and X_recon (if self.reconstruct=True)
        """
        from tensorly.decomposition import tensor_train
        if len(self.X.shape)==4:
            self.X = self.X.permute(0,2,3,1) # put the channel at the end
        factors = tensor_train(self.X,rank=self.ranks,verbose=self.verbose)
        if self.reconstruct:
            from tensorly import tt_to_tensor
            X_recon = tt_to_tensor(factors)
            err = self.relative_error(self.X,X_recon)
            if len(self.X.shape) == 4:
                X_recon = X_recon.permute(0,3,1,2) # put the channel back to the right place
            if self.verbose: print('Reconstruction Error::',err)
            return factors,X_recon,err
        else:
            return factors

    def NNSVD(self):
        r"""
        Performs NNSVD.

        Returns
        -------
        tuple
            A tuple containing:
            - factors (list): a list of factor matrices
            - X_recon (torch.Tensor): the reconstructed tensor (if self.reconstruct=True)
            - err (float): the relative error between X and X_recon (if self.reconstruct=True)
        """
        if self.reconstruct:
            (U,S,V),_,_ = self.SVD()
        else:
            (U, S, V)  = self.SVD()
        UP = torch.where(U > 0, U, 0)
        UN = torch.where(U < 0, -U, 0)
        VP = torch.where(V > 0, V, 0)
        VN = torch.where(V < 0, -V, 0)
        eps= torch.finfo(U.dtype).eps
        UP_norm = torch.sqrt(torch.sum(torch.square(UP), 0))
        UN_norm = torch.sqrt(torch.sum(torch.square(UN), 0))
        VP_norm = torch.sqrt(torch.sum(torch.square(VP), 0))
        VN_norm = torch.sqrt(torch.sum(torch.square(VN), 0))
        mp = torch.sqrt(UP_norm * VP_norm * S)
        mn = torch.sqrt(UN_norm * VN_norm * S)
        W = torch.where(mp > mn, mp * UP / (UP_norm + eps), mn * UN / (UN_norm + eps))
        H = torch.where(mp > mn, mp * VP / (VP_norm + eps), mn * VN / (VN_norm + eps)).T
        Wsum = torch.sum(W, 0, keepdims=True)
        H = H * Wsum.T
        W = W / torch.clamp(Wsum, min=eps)
        if self.reconstruct==True:
            X_recon = W@H
            err = self.relative_error(self.X,X_recon)
            if self.verbose: print('Reconstruction Error::',err)
            return (W, H), X_recon,err
        else:
            return (W, H)

    def SVD(self):
        r"""
        Performs SVD.

        Returns
        -------
        tuple
            A tuple containing:
            - factors (list): a list of factor matrices
            - X_recon (torch.Tensor): the reconstructed tensor (if self.reconstruct=True)
            - err (float): the relative error between X and X_recon (if self.reconstruct=True)
        """
        U, S, V =torch.svd(self.X)
        k = self.ranks
        U, S, V = U[:, :k], S[:k], V[:, :k].T
        if self.reconstruct:
            X_recon = U@torch.diag(S)@V
            err = self.relative_error(self.X,X_recon)
            if self.verbose: print('Reconstruction Error::',err)
            return (U,S,V.T),X_recon,err
        else:
            return (U,S,V.T)

    def NMF(self):
        r"""
        Performs NMF.

        Returns
        -------
        tuple
            A tuple containing:
            - factors (list): a list of factor matrices
            - X_recon (torch.Tensor): the reconstructed tensor (if self.reconstruct=True)
            - err (float): the relative error between X and X_recon (if self.reconstruct=True)
        """
        m,n = self.X.shape
        if self.params['init']=='rand':
            Winit,Hinit = torch.random(m,self.ranks),torch.random(self.ranks,n)
        elif self.params['init']=='nnsvd':
            if self.reconstruct:
                (Winit,Hinit),_,_ = self.NNSVD()
            else:
                (Winit,Hinit) = self.NNSVD()
        err = []
        Wcur=Winit
        Hcur=Hinit
        for i in range(self.params['max_iter']):
            #Wupdate
            Wcur = torch.mul(Wcur,torch.div(torch.matmul(self.X,Hcur.T),torch.matmul(Wcur,torch.matmul(Hcur,Hcur.T))))
            #Hupdate
            Hcur = torch.mul(Hcur,torch.div(torch.matmul(Wcur.T,self.X),torch.matmul(Wcur.T,torch.matmul(Wcur,Hcur))))
            err.append(self.relative_error(self.X,Wcur@Hcur))
            if i>0:
                if abs(err[-1]-err[-2])<self.params['tol']:
                    break
        if self.verbose: print('performed iterations=',i)
        return (Wcur,Hcur),Wcur@Hcur,err[-1]

    def relative_error(self,data,data_recon):
        r"""
        Calculates the relative error between two tensors data and data_recon.

        Parameters
        ----------
        data : torch.Tensor
            The original tensor.
        data_recon : torch.Tensor
            The reconstructed tensor.

        Returns
        -------
        float
            The relative error between data and data_recon.
        """
        return torch.linalg.norm(data-data_recon)/torch.linalg.norm(data)

    def ttm(self,X,v,mode,transpose=False):
        r"""
        Performs tensor-time-matrix multiplication along a given mode using the tensorly library.

        Parameters
        ----------
        X : torch.Tensor
            The tensor to be multiplied.
        v : torch.Tensor or list of torch.Tensor
            The matrix or matrices to multiply with.
        mode : int or list of int
            The modes in which to multiply the tensors.
        transpose : bool, optional
            Whether or not to transpose the matrix before multiplication.

        Returns
        -------
        torch.Tensor
            The tensor after performing the tensor-time-matrix multiplication.
        """
        from tensorly.tenalg import multi_mode_dot,mode_dot
        if  isinstance(mode,int):
            return mode_dot(X, v, mode=mode, transpose=transpose)
        else:
            return multi_mode_dot(X, v, modes=mode, skip=None, transpose=transpose)

    def fit(self):
        r"""
        Fits the decomposition model to the input tensor X using various decomposition methods supported by tensorly.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor to fit.

        Returns
        -------
        tuple
            A tuple containing:
            - X_recon (torch.Tensor): the reconstructed tensor
            - err (float): the relative error between X and X_recon
        """
        if self.method=='tucker':
            _,X_recon,err = self.tucker()
        elif self.method=='ntucker':
            _,X_recon,err = self.ntucker()
        elif self.method == 'cpd':
            _,X_recon,err = self.cpd()
        elif self.method == 'ncpd':
            _,X_recon,err = self.ncpd()
        elif self.method == 'tt':
            _,X_recon,err = self.tt()
        elif self.method == 'nnsvd':
            _,X_recon,err = self.NNSVD()
        elif self.method == 'svd':
            _,X_recon,err = self.SVD()
        elif self.method == 'nmf':
            _,X_recon,err = self.NMF()
        else:
            raise NotImplementedError
        return X_recon,err

    def forward(self,X,ranks):
        r"""
        Performs forward pass through the model for both batch and single mode.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, num_items).
        ranks : list
            List of rank values for each tensor in the CP decomposition.

        Returns
        -------
        tuple
            A tuple containing:
            - X_recon (torch.Tensor): the reconstructed tensor of shape (batch_size, num_items)
            - err (float): the reconstruction error between input tensor and reconstructed tensor
        """
        #flag = 0
        self.ranks = ranks
        if self.data_mode=='batch':
            self.X = X
            X_recon,err = self.fit()
        elif self.data_mode == 'single':
            self.X = X
            if self.method=='tucker':
                X_recon,err = self.fit()
            else:
                tmp = [self.fit() for self.X in X]
                X_recon = torch.stack([i[0] for i in tmp])
                err = torch.stack([i[1] for i in tmp])
        return X_recon,err
