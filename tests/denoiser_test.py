import sys
import torch
import time
from AdversarialTensors.denoiser import Denoiser


data_modes = ['batch','single']

data = torch.rand(100,3,32,32)
if __name__ == '__main__':

    for data_mode in data_modes:
        t1 = time.time()
        denoiser = Denoiser(method='tucker', device=0,
                            tensor_params={'factors': None, 'init': 'svd', 'tol': 1e-5, 'svd':'truncated_svd','max_iter': 100},
                            verbose=True, patch_params={'patch_size': 8, 'stride': 4, 'channels': 3}, data_mode=data_mode)
        if data_mode == 'single':
            ranks = [12,3,4,4]
        else:
            ranks = [8,12,3,4,4]
        res,err  = denoiser(data,ranks,recon_err=True)
        t2 = time.time()
        print('Done for data mode',data_mode,'in ',t2-t1,' time')
