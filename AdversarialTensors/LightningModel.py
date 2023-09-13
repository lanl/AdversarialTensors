from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.optim as optim
from AdversarialTensors.models.resnet import resnet18, resnet34, resnet50
from AdversarialTensors.models.resnet_orig import OrigResNet18, OrigResNet34, OrigResNet50
from AdversarialTensors.models.wide_resnet import WideResNet
import math
import sys
from .adv_attacks import Attacks
from .normalize import Normalize
from .denoiser import Denoiser
# from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py

def learning_rate_cifar10(init, epoch):
    r"""
    Computes the learning rate for CIFAR-10 training based on the initial learning rate and the current epoch.

    Parameters
    ----------
    init : float
        The initial learning rate at epoch 0.
    epoch : int
        The current training epoch.

    Returns
    -------
    float
        The adjusted learning rate for the current epoch.

    Notes
    -----
    This function computes the learning rate based on a piecewise schedule.
    It divides the training into 4 phases and scales the learning rate by 0.2 raised to a factor.
    The factor is determined by the current epoch:
        - 0-60:   factor = 0
        - 61-120: factor = 1
        - 121-160: factor = 2
        - 161-∞: factor = 3
    """

    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

class LightningModel(LightningModule):
    r"""
    PyTorch Lightning Module for training various ResNet architectures and their Wide ResNet and Original variants.

    Attributes
    ----------
    model_name : str
        The name of the model architecture to use.
    lr : float, default=1e-3
        Learning rate.
    batch_size : int, default=1
        Batch size for training and evaluation.
    num_batches : int, default=1
        Number of batches per epoch.
    num_classes : int, default=10
        Number of output classes.
    adv_settings : dict, optional
        Settings for adversarial attack generation. If None, no attacks are used.
    normalize_settings : dict, optional
        Settings for input normalization. If None, no normalization is applied.
    denoiser_settings : dict, optional
        Settings for denoising the inputs. If None, no denoising is applied.
    ...

    Methods
    -------
    forward(x: torch.Tensor):
        Forward pass through the model.

    on_train_start():
        Actions to perform when training starts.

    generate_attack(batch: Any):
        Generate adversarial attack samples.

    model_step(batch: Any):
        Defines a single step during model training.

    training_step(batch: Any, batch_idx: int):
        Defines a single step during training loop.

    on_train_epoch_end():
        Actions to perform at the end of each training epoch.

    validation_step(batch: Any, batch_idx: int):
        Defines a single step during validation loop.

    on_validation_epoch_end():
        Actions to perform at the end of each validation epoch.

    test_step(batch: Any, batch_idx: int):
        Defines a single step during testing loop.

    on_test_epoch_end():
        Actions to perform at the end of each test epoch.

    resnet_configure_optimizers():
        Configure optimizers and learning rate schedules for ResNet models.

    wide_resnet_configure_optimizers():
        Configure optimizers and learning rate schedules for Wide ResNet models.

    configure_optimizers():
        General configuration of optimizers based on model types.

    Notes
    -----
    This class integrates with PyTorch Lightning to enable training, validation, and testing loops,
    and also allows for adversarial attack generation, normalization, and denoising.

    Example
    -------
    model = LightningModel('resnet18', lr=0.001)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model)

    """

    def __init__(
        self,
        model_name,
        lr = 1e-3,
        batch_size = 1,
        num_batches = 1,
        num_classes = 10,
        adv_settings = None,
        normalize_settings = None,
        denoiser_settings = None,
        *args, **kwargs
    ):
        super().__init__()
        """
        Initialize the LightningModel with given settings.
        """

        self.save_hyperparameters()
        self.model_name = model_name
        if model_name == "resnet18":
            self.model = resnet18(num_classes=num_classes)
        elif model_name == "resnet34":
            self.model = resnet34(num_classes=num_classes)
        elif model_name == "resnet50":
            self.model = resnet50(num_classes=num_classes)
        elif model_name == "origresnet18":
            self.model = OrigResNet18(num_classes=num_classes)
        elif model_name == "origresnet34":
            self.model = OrigResNet34(num_classes=num_classes)
        elif model_name == "origresnet50":
            self.model = OrigResNet50(num_classes=num_classes)
        elif model_name == "wresnet28_10":
            self.model = WideResNet(depth=28, widen_factor=10, dropRate=0.3, sub_block1=True, num_classes=num_classes)
        else:
            sys.exit(f'Unrecognized model name ({model_name})!')


        # no-op generator
        if adv_settings == None:
            self.adv_mode = False
        else:
            self.adv_mode = True
            #self.automatic_optimization = False
            self.adv_gen = Attacks(self.model, **adv_settings)

        if denoiser_settings != None:
            denoiser = Denoiser(**denoiser_settings)
            self.model = torch.nn.Sequential(
                denoiser,
                self.model,
            )

        # no-op generator
        if normalize_settings != None:
            self.model = torch.nn.Sequential(
                Normalize(normalize_settings['mean'], normalize_settings['std']),
                self.model,
            )

        self.lr = lr
        # these are only useful for the scheduler
        self.num_batches = num_batches
        self.batch_size = batch_size

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        if self.adv_mode:
            # metric objects for calculating and averaging accuracy across batches
            self.val_adv_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_adv_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.train_adv_acc = Accuracy(task="multiclass", num_classes=num_classes)
            # for averaging loss across batches
            self.train_adv_loss = MeanMetric()
            self.val_adv_loss = MeanMetric()
            self.test_adv_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        r"""
        Perform forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.model(x)

    def on_train_start(self):
        r"""
        Reset all validation metrics before the training starts.

        This is necessary because PyTorch Lightning runs validation sanity checks
        before the actual training, and we don't want these checks to affect our metrics.
        """
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        if self.adv_mode:
            self.val_adv_loss.reset()
            self.val_adv_acc.reset()

    def generate_attack(self, batch):
        r"""
        Generate adversarial tests for a given batch.

        Parameters
        ----------
        batch : tuple
            A tuple containing input tensor `x` and target tensor `y`.

        Returns
        -------
        tuple
            A tuple containing adversarial tests `adv_x` and the target tensor `y`.
        """
        with torch.enable_grad():
            x, y = batch
            adv_x = self.adv_gen.generate(x, y)
        #self.model.zero_grad()
        return (adv_x, y)

    def model_step(self, batch: Any):
        r"""
        Forward pass through the model and compute loss and predictions.

        Parameters
        ----------
        batch : Any
            The input batch data.

        Returns
        -------
        tuple
            A tuple containing the loss tensor, predicted labels, and ground truth labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        r"""
        Perform a single training step.

        Parameters
        ----------
        batch : Any
            The input batch data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The training loss.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.adv_mode:
            batch = self.generate_attack(batch)
            loss_adv, preds, targets = self.model_step(batch)

            # update and log loss_adv
            self.train_adv_loss(loss)
            self.train_adv_acc(preds, targets)
            self.log("train_adv_loss", self.train_adv_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_adv_acc", self.train_adv_acc, on_step=False, on_epoch=True, prog_bar=True)
            loss = loss + loss_adv
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        r"""
        Operations to perform at the end of each training epoch.

        Currently a placeholder with no actions.
        """
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Perform a single validation step.

        Parameters
        ----------
        batch : Any
            The input batch data.
        batch_idx : int
            The index of the current batch.

        """
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.adv_mode:
            adv_batch = self.generate_attack(batch)
            loss_adv, preds_adv, targets_adv = self.model_step(adv_batch)
            self.val_adv_loss(loss_adv)
            self.val_adv_acc(preds_adv, targets_adv)
            self.log("val_adv_loss", self.val_adv_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_adv_acc", self.val_adv_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        r"""
        Operations to perform at the end of each validation epoch.

        Computes and logs the best validation accuracy achieved so far.
        """
        acc = self.val_acc.compute()  # get current val acc
        adv_acc = acc
        if self.adv_mode:
            adv_acc = self.val_adv_acc.compute()  # get current val acc
        self.val_acc_best((acc + adv_acc) / 2.0)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        r"""
        Perform a single test step.

        Parameters
        ----------
        batch : Any
            The input batch data.
        batch_idx : int
            The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.adv_mode:
            adv_batch = self.generate_attack(batch)
            loss_adv, preds_adv, targets_adv = self.model_step(adv_batch)
            self.test_adv_loss(loss_adv)
            self.test_adv_acc(preds_adv, targets_adv)
            self.log("test_adv_loss", self.test_adv_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_adv_acc", self.test_adv_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        r"""
        Operations to perform at the end of each test epoch.

        Currently a placeholder with no actions.
        """
        pass

    def resnet_configure_optimizers(self):
        r"""
        Configure optimizers and learning rate schedulers for ResNet architectures.

        Returns
        -------
        dict
            A dictionary containing the optimizer and learning rate scheduler.
        """
        # from: https://github.com/huyvnphan/PyTorch_CIFAR10
        from scheduler import WarmupCosineLR

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                   momentum=0.9, weight_decay=1e-2,
                                   nesterov=True)
        total_steps = self.trainer.max_epochs * self.num_batches
        scheduler = WarmupCosineLR(optimizer, warmup_epochs=total_steps * 0.3,
                                   max_epochs=total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    def wide_resnet_configure_optimizers(self):
        r"""
        Configure optimizers and learning rate schedulers for Wide ResNet architectures.

        Returns
        -------
        dict
            A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                   momentum=0.9, weight_decay=5e-4,
                                   nesterov=True)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: learning_rate_cifar10(self.lr, epoch))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                #"monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


    def configure_optimizers(self):
        r"""
        Configure optimizers and learning rate schedulers based on the model architecture.

        Examples and more info:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns
        -------
        dict or list
            The configuration for the optimizer and learning rate scheduler.
        """
        if self.model_name.startswith('resnet') or self.model_name.startswith('origresnet'):
            return self.resnet_configure_optimizers()
        else:
            return self.wide_resnet_configure_optimizers()
