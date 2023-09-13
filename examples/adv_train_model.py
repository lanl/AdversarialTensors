#@author: Manish Bhattarai, Mehmet Cagri
import torch
from AdversarialTensors.datasets import DatasetLoader
from AdversarialTensors.utils import eval_accuracy_dataloader, eval_accuracy_dataloader_with_attack
import numpy as np
from AdversarialTensors.my_progress_bar import MyProgressBar
from AdversarialTensors.LightningModel import LightningModel
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
from foolbox import PyTorchModel
from AdversarialTensors.adv_attacks import Attacks

if __name__ == '__main__':
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description='Adversarial Model Trainer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Define all the command-line arguments
    # Add comments to clarify what each argument is supposed to do

    # Initialize learning rate
    parser.add_argument('--init_lr', metavar='init_lr', type=float, default=1e-1, help='Initial learning rate')
    # Batch size for training
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=128, help='Batch size for training')
    # Model architecture to be used
    parser.add_argument('--model_name', metavar='model_name', default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', "wresnet28_10"],
                        help='Name of the model architecture')
    # Maximum number of epochs for training
    parser.add_argument('--max_epoch', metavar='max_epoch', type=int, default=200,
                        help='Maximum number of epochs for training')
    # Dataset to be used for training and testing
    parser.add_argument('--dataset_name', metavar='dataset_name', default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', "fmnist", "imagenet"], help='Name of the dataset')
    # Number of folds for cross-validation
    parser.add_argument('--nfolds', metavar='nfolds', type=int, default=8, help='Number of folds for cross-validation')
    # Index of the current fold in cross-validation
    parser.add_argument('--fold_ind', metavar='fold_ind', type=int, default=0,
                        help='Index of the current fold in cross-validation')
    # Device ID for CUDA
    parser.add_argument('--device_id', metavar='device_id', type=int, default=0, help='CUDA device ID')
    # Whether to use a denoising model
    parser.add_argument('--use_denoiser', metavar='use_denoiser', type=int, default=0, choices=[0, 1],
                        help='Use denoiser')
    # Whether to train the model with adversarial training
    parser.add_argument('--adv_train', metavar='adv_train', type=int, default=0, choices=[0, 1],
                        help='Use adversarial training')

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    # Initial setup
    device = f"cuda:{args.device_id}"
    batch_size = args.batch_size
    lr = args.init_lr
    model_name = args.model_name
    nfolds = args.nfolds
    dataset_name = args.dataset_name
    fold_ind = args.fold_ind
    max_epoch = args.max_epoch

    # Seed setting
    pl.seed_everything(fold_ind, workers=True)

    # Data Loading
    # Setup data loader based on the given dataset
    data_params = {'batch_size': batch_size, 'num_workers': 4, 'shuffle': True, 'normalize': False, 'nfolds': nfolds}
    dataloader = DatasetLoader(name=dataset_name, params=data_params)
    splits, testloader, classes, mean, std = dataloader.fit()

    # Split the dataset for cross-validation
    my_training_data = splits[fold_ind][0]
    my_valid_data = splits[fold_ind][1]

    # Optional settings for denoiser and adversarial training
    denoiser_settings = None
    adv_settings = None
    if args.use_denoiser:
        denoiser_settings = {'method': 'tt', 'device': device,
                             'tensor_params': {'factors': None, 'init': 'svd', 'svd': 'truncated_svd', 'tol': 1e-5,
                                               'max_iter': 1},
                             'verbose': False,
                             'patch_params': {'patch_size': 8, 'stride': 4, 'channels': 3, 'padding': 1},
                             'data_mode': 'single',
                             'ranks': [1, 12, 6, 3, 1]}
    if args.adv_train:
        adv_settings = {'attack': 'fgsm', 'eps': 8 / 255, 'device': 0}

    # Initialize the model
    model = LightningModel(model_name=model_name, lr=lr,
                           batch_size=batch_size,
                           num_batches=len(my_training_data),
                           num_classes=len(classes),
                           adv_settings=adv_settings,
                           normalize_settings={'mean': mean, 'std': std},
                           denoiser_settings=denoiser_settings)

    # Logger setup for saving training metrics
    logger_folder = f"adv_log_{dataset_name}_{model_name}"
    log_sub_folder = f"{fold_ind}_{nfolds}"
    logger = CSVLogger(logger_folder, name=log_sub_folder)

    # Callbacks and Trainer Initialization
    bar = MyProgressBar(refresh_rate=1)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=9999, verbose=False, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1,
                                          dirpath=f"my_adv_checkpoints/{dataset_name}/",
                                          filename=f"{model_name}_{fold_ind}_{nfolds}")

    # Initialize the trainer and start training
    trainer = pl.Trainer(accelerator='gpu', inference_mode=False, devices=[args.device_id, ], max_epochs=max_epoch,
                         logger=logger, callbacks=[bar, early_stop_callback, checkpoint_callback])
    trainer.fit(model, my_training_data, my_valid_data)

    # Load the best model and evaluate
    best_model_path = trainer.checkpoint_callback.best_model_path
    print("best model path:", best_model_path)
    best_lightning_model = LightningModel.load_from_checkpoint(best_model_path).eval()
    best_torch_model = best_lightning_model.model

    # Evaluate clean and adversarial accuracy
    acc = eval_accuracy_dataloader(best_torch_model, testloader)
    print("clean accuracy on test data: ", acc)
    attacker = Attacks(model=best_torch_model, attack='fgsm', eps=8 / 255, norm='Linf', device=0, bounds=(0, 1))
    adv_acc = eval_accuracy_dataloader_with_attack(best_torch_model, testloader, attacker)
    print("Adv. accuracy on test data: ", adv_acc)
