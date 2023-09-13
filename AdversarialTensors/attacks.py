'''
We will utilize the models and attacks from
https://robustbench.github.io/
pip install -q foolbox
pip install autoattack
'''
from autoattack import AutoAttack
import foolbox.attacks as fa
from foolbox import PyTorchModel
import torch
import torch
import numpy as np
from .utils import *

class Attacks(torch.nn.Module):
    """
    Initializes the Attacks class.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model that takes image tensors as input.
    attack : str, optional
        The type of attack to perform. Options include 'all', 'fgsm', 'pgd', 'bim', 'uniform', 'deepfool', 'autoattack'. Default is 'all'.
    attack_params : dict, optional
        Additional parameters for the attack method.
        Includes keys for 'norm', 'eps', 'version', 'log_dir', 'seed', and 'exp'.
    device : int, optional
        The device to perform the attack on. Default is 0.

    """
    def __init__(self,model=None,attack='all',attack_params={'norm':2,'eps':8/255,'version':'standard','log_dir':'autoattack/','seed':99,'exp':'all'},device=0):

        super(Attacks, self).__init__()
        self.model = model
        self.attack = attack
        self.attack_params = attack_params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.attack_params['eps'] is not None:
            self.eps = self.attack_params['eps']
        else:
            self.eps = 2 ** np.arange(9) / 256

    def init_attack(self):
        """
        Initializes the attack model based on the chosen type of attack.

        """
        self.models = PyTorchModel(self.model, bounds=(0, 1))
        if self.attack == 'autoattack':
            # attack version can be standard/plus/rand for autoattack
            self.attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            try:
                import os
                os.mkdir(self.attack_params['log_dir'])
            except:
                pass
            self.adversary_model = AutoAttack(self.models, norm=self.attack_params['norm'], eps=self.eps,
                                          version=self.attack_params['version'], attacks_to_run=self.attack_list,
                                          log_path=self.attack_params['log_dir']+'/log_resnet.txt', device=self.device)
            self.adversary_model.apgd.n_restarts = 1
            self.adversary_model.fab.n_restarts = 1
            self.adversary_model.apgd_targeted.n_restarts = 1
            self.adversary_model.fab.n_target_classes = 9
            self.adversary_model.apgd_targeted.n_target_classes = 9
            self.adversary_model.square.n_queries = 5000

        elif self.attack == 'fgsm':
            self.adversary_model = [fa.FGSM()]
        elif self.attack == 'pgd':
            self.adversary_model = [fa.LinfPGD()]
        elif self.attack == 'bim':
            self.adversary_model = [fa.LinfBasicIterativeAttack()]
        elif self.attack == 'uniform':
            self.adversary_model = [fa.LinfAdditiveUniformNoiseAttack()]
        elif self.attack == 'deepfool':
            self.adversary_model = [fa.LinfDeepFoolAttack()]
        elif self.attack == 'all':
            self.adversary_model = [
                fa.FGSM(),
                fa.LinfPGD(),
                fa.LinfBasicIterativeAttack(),
                fa.LinfAdditiveUniformNoiseAttack(),
                fa.LinfDeepFoolAttack()]




    def attack_eval_all(self,X,y,batch=128,verbose=True):
        """
        Evaluate the adversarial robustness of the model against all specified attacks on a given dataset.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (N, C, H, W).
        y : numpy.ndarray
            Ground-truth labels for the input data of shape (N,).
        batch : int, optional
            Batch size for generating adversarial tests. Default is 128.
        verbose : bool, optional
            Whether to print verbose output. Default is True.

        Returns
        -------
        x_adv_resnet : list
            A list of adversarial tests generated for each attack method.

        """
        self.init_attack()
        from foolbox import  accuracy, samples
        clean_acc = accuracy(self.models, X, y)
        print(f"clean accuracy:  {clean_acc * 100:.1f} %")
        print("")
        if self.attack == 'autoattack':
            x_adv_resnet = self.adversary_model.run_standard_evaluation(X, y, bs=batch)
            print(f'x_adv_resnet shape: {x_adv_resnet.shape}')
            #torch.save([x_adv_resnet, y], f'{self.attack_params.log_dir}/x_adv_resnet_sd{self.attack_params.seed}.pt')


        else: #if self.attacks == 'all':
            if verbose: attack_success = np.zeros((len(self.adversary_model), len(self.eps), len(X)), dtype=np.bool)
            x_adv_resnet = [None]*len(self.adversary_model)
            for i, attack in enumerate(self.adversary_model):
                _, x_adv_resnet[i], success = attack(self.models, X, y, epsilons=self.eps)
                if verbose:
                    assert success.shape == (len(self.eps), len(X))
                    success_ = success.numpy()
                    assert success_.dtype == np.bool
                    attack_success[i] = success_
                    print(attack)
                    print("  ", 1.0 - success_.mean(axis=-1).round(2))

            if verbose:
                # calculate and report the robust accuracy (the accuracy of the model when
                # it is attacked) using the best attack per sample
                robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
                print("")
                print("-" * 79)
                print("")
                print("worst case (best attack per-sample)")
                print("  ", robust_accuracy.round(2))
                print("")

                print("robust accuracy for perturbations with")
                for eps, acc in zip(self.eps, robust_accuracy):
                    print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        return x_adv_resnet



    def attack_eval_dataloader(self,dataloader,batch=128,verbose=True):
        """
        Evaluate the adversarial robustness of the model on a given dataset using a PyTorch DataLoader.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            A PyTorch DataLoader object.
        batch : int, optional
            Batch size for generating adversarial tests. Default is 128.
        verbose : bool, optional
            Whether to print verbose output. Default is True.

        Returns
        -------
        clean_acc : float
            Clean accuracy of the model.
        acc : float
            Adversarial accuracy of the model.

        """
        self.init_attack()
        from foolbox import  accuracy, samples

        clean_acc = eval_accuracy_dataloader(self.models, dataloader, self.device)
        if verbose:
            print(f"clean accuracy:  {clean_acc * 100:.1f} %")
            print("")
        total = 0
        correct = 0
        if self.attack == 'autoattack':
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                x_adv_resnet = self.adversary_model.run_standard_evaluation(inputs, targets, bs=len(targets))
                acc,correct,total = eval_accuracy(self.models, x_adv_resnet, targets,correct,total)
            if verbose:
                print('Advarasarial accuracy:',acc)
                #print(f'x_adv_resnet shape: {x_adv_resnet.shape}')
                #torch.save([x_adv_resnet, y], f'{self.attack_params.log_dir}/x_adv_resnet_sd{self.attack_params.seed}.pt')


        else: #if self.attacks == 'all':
            #if verbose: attack_success = np.zeros((len(self.attacks), len(self.eps), len(X)), dtype=np.bool)
            #accuracy = np.zeros((len(self.adversary_model),len(self.eps)))
            x_adv_resnet = [None]*len(self.adversary_model)
            for i, attack in enumerate(self.adversary_model):
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    res = attack(self.models, inputs, targets, epsilons=self.eps)
                    _, x_adv_resnet, success = attack(self.models, inputs, targets, epsilons=self.eps)
                    acc, correct, total = eval_accuracy(self.models, x_adv_resnet, targets, correct, total)
                #accuracy[i]=acc

                if verbose:
                    print('Advarasarial accuracy:', acc)
        return clean_acc, acc


    def forward(self,X,y):
        """
        Generates adversarial tests for a given batch of images.

        Parameters
        ----------
        X : torch.Tensor
            Input images of shape (batch_size, C, H, W).
        y : torch.Tensor
            Ground-truth labels of shape (batch_size,).

        Returns
        -------
        x_adv_resnet : torch.Tensor
            Adversarial tests generated for each attack method.

        """
        self.init_attack()
        if self.attack == 'autoattack':
            x_adv_resnet = self.adversary_model.run_standard_evaluation(X, y, bs=len(y))
            return x_adv_resnet
        else:
            x_adv_resnet = [None]*len(self.adversary_model)
            for i,adv_model in enumerate(self.adversary_model):
                _, x_adv_resnet[i], success = adv_model(self.models, X,y, epsilons=self.eps)
            return x_adv_resnet[0]
        #return  x_adv_resnet
