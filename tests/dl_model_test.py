import sys
import torch
import time

from AdversarialTensors.model import FinalModel
from AdversarialTensors.models.resnet import resnet18
from AdversarialTensors.denoiser import Denoiser



data_modes = ['batch','single']

data = torch.rand(100,3,32,32)
labels = (torch.rand(size=(100,10)) < 0.25).int().argmax(axis=1)
dl_model = resnet18(pretrained=False, progress=True, num_classes=100)
if __name__ == '__main__':

    for data_mode in data_modes:
        t1 = time.time()

        if data_mode == 'single':
            ranks = [12,3,4,4]
        else:
            ranks = [8,12,3,4,4]
        denoiser = None
        denoiser =  Denoiser(method='tucker', device=0,
                            tensor_params={'factors': None, 'init': 'svd', 'tol': 1e-5, 'max_iter': 100,'svd':'truncated_svd'},
                            verbose=True, patch_params={'patch_size': 8, 'stride': 4, 'channels': 3},
                            data_mode=data_mode)


        model = FinalModel(model=dl_model, denoiser=denoiser)
        model.train()
        data.requires_grad = True
        outputs = model(data, ranks)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs,labels)
        loss.backward()
        gradient = data.grad
        print(gradient.max(),gradient.min(),torch.norm(gradient))
        t2 = time.time()
        print('Done for data mode',data_mode,'in ',t2-t1,' time')
