import torch
import time

from AdversarialTensors.model import FinalModel
from AdversarialTensors.models.resnet import resnet18
from AdversarialTensors.denoiser import Denoiser
from AdversarialTensors.attacks import Attacks



attack_modes = ['fgsm','all','autoattack']

data = torch.rand(100,3,32,32)
labels = (torch.rand(size=(100,10)) < 0.25).int().argmax(axis=1)
dl_model = resnet18(pretrained=False, progress=True, num_classes=100)
if __name__ == '__main__':

    for attack_mode in attack_modes:
        print('working on attack mode=',attack_mode)
        t1 = time.time()

        ''' if data_mode == 'single':
            ranks = [12,3,4,4]
        else:
            ranks = [8,12,3,4,4]
        denoiser = None'''
        data_mode = 'batch'
        ranks = [8, 12, 3, 4, 4]
        denoiser =  Denoiser(method='tucker', device=0,
                            tensor_params={'factors': None, 'init': 'svd', 'tol': 1e-5, 'max_iter': 100},
                            verbose=False, patch_params={'patch_size': 8, 'stride': 4, 'channels': 3},
                            data_mode=data_mode,ranks=ranks)
        #denoiser = None
        if attack_mode == 'autoattack':
            norm = 'L2'
        else:
            norm = 2

        model = FinalModel(model=dl_model, denoiser=denoiser)
        attacker = Attacks(model, attack = attack_mode, attack_params = {'norm':norm, 'eps': 8 / 255, 'version': 'custom', 'log_dir': 'autoattack/',
                                                'seed': 99, 'exp': 'all'}, device = 0)

        outputs = attacker(data,labels)


        t2 = time.time()
        print('Done for attack mode',attack_mode,'in ',t2-t1,' time')
