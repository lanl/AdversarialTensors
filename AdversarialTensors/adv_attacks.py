#@author: Manish Bhattarai, Mehmet Cagri
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
import torchattacks


class Attacks():
    def __init__(self,model=None,attack='fgsm', eps=8/255, norm = 'Linf', device=0, bounds=(0,1)):
        """
        Initializes the Attacks class.

        Parameters
        ----------
        model : torch.nn.Module
            A PyTorch model that takes image tensors as input.
        attack : str, optional
            The type of attack to perform. Default is 'fgsm'.
        eps : float, optional
            The maximum allowed perturbation to the image. Default is 8/255.
        norm : str, optional
            The Lp-norm used in the attack. Default is 'Linf'.
        device : int, optional
            The device to perform the attack on. Default is 0.
        bounds : tuple, optional
            The minimum and maximum values for the image tensor. Default is (0, 1).

        """

        self.model = PyTorchModel(model, bounds=bounds, device=device)
        self.attack = attack
        self.device = device
        self.eps = eps
        self.norm = norm
        self.bounds = bounds

        self.init_attack()

    def init_attack(self):
        """Initializes the attack based on the provided attack type."""

        if self.attack == 'fgsm':
            self.adversary_model = fa.FGSM()
        elif self.attack == 'pgd':
            self.adversary_model = fa.LinfPGD(steps=10)
        elif self.attack == 'bim':
            self.adversary_model = fa.LinfBasicIterativeAttack()
        elif self.attack == 'uniform':
            self.adversary_model = fa.LinfAdditiveUniformNoiseAttack()
        elif self.attack == 'deepfool':
            self.adversary_model = fa.LinfDeepFoolAttack()


    def generate(self, X, y):
        """
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
        #model = PyTorchModel(model.eval(), bounds=self.bounds, device=self.device)
        _, x_adv_resnet, success = self.adversary_model(self.model, X,y, epsilons=self.eps)
        #return X
        return  x_adv_resnet
