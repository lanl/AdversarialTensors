#@author: Manish Bhattarai, Mehmet Cagri

from autoattack import AutoAttack
import foolbox.attacks as fa
from foolbox import PyTorchModel
import torch
import torch
import numpy as np
from .utils import *

class Attacks(torch.nn.Module):
    r"""
    A PyTorch Module to perform adversarial attacks on a given model.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model that takes image tensors as input.
    attack : str, optional
        The type of attack to be performed. Default is 'all'.
    attack_params : dict, optional
        Additional parameters for the attack. Default includes L2 norm, epsilon value, etc.
    device : str, optional
        The computation device. Default is 'cuda'.

    Attributes
    ----------
    model_wrapper : PyTorchModel
        Wrapped version of the PyTorch model compatible with Foolbox.
    attack_list : list of str
        List of attacks to be used when using AutoAttack.
    adversary_model : object
        Foolbox attack object.
    """
    def __init__(self,model=None,attack='all',attack_params={'norm':2,'eps':8/255,'version':'standard','log_dir':'autoattack/','seed':99,'exp':'all'},device='cuda'):

        super(Attacks, self).__init__()
        self.model = model
        self.model_wrapper = PyTorchModel(self.model, bounds=(0, 1), device=device)
        self.attack = attack
        self.attack_params = attack_params
        self.device = device
        self.eps = self.attack_params['eps']
        self.init_attack()

    def init_attack(self):
        r"""
        Initializes the Foolbox attack model based on the type of attack selected.
        """
        if self.attack == 'autoattack':
            # attack version can be standard/plus/rand for autoattack
            self.attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            self.adversary_model = AutoAttack(self.model, norm=self.attack_params['norm'], eps=self.eps,
                                          version=self.attack_params['version'], attacks_to_run=self.attack_list,
                                          log_path=self.attack_params['log_dir']+'/log_resnet.txt', device=self.device)
            self.adversary_model.apgd.n_restarts = 1
            self.adversary_model.fab.n_restarts = 1
            self.adversary_model.apgd_targeted.n_restarts = 1
            self.adversary_model.fab.n_target_classes = 9
            self.adversary_model.apgd_targeted.n_target_classes = 9
            self.adversary_model.square.n_queries = 5000

        elif self.attack == 'fgsm':
            self.adversary_model = fa.FGSM()
        elif self.attack == 'pgd':
            self.adversary_model = fa.LinfPGD()
        elif self.attack == 'bim':
            self.adversary_model = fa.LinfBasicIterativeAttack()
        elif self.attack == 'uniform':
            self.adversary_model = fa.LinfAdditiveUniformNoiseAttack()
        elif self.attack == 'deepfool':
            self.adversary_model = fa.LinfDeepFoolAttack()

    def forward(self,X,y):
        r"""
        Executes the attack on the given inputs and labels.

        Parameters
        ----------
        X : torch.Tensor
            Input images.
        y : torch.Tensor
            True labels.

        Returns
        -------
        x_adv_resnet : torch.Tensor
            Adversarially perturbed images.
        """
        if self.attack == 'autoattack':
            x_adv_resnet = self.adversary_model.run_standard_evaluation(X, y, bs=len(y))
            return x_adv_resnet
        else:
            _, x_adv_resnet, success = self.adversary_model(self.model_wrapper, X, y, epsilons=self.eps)
            return x_adv_resnet
