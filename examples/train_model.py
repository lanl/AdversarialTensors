#@author:  Mehmet Cagri

import torch
from AdversarialTensors.datasets import DatasetLoader
from AdversarialTensors.utils import eval_accuracy_dataloader
import numpy as np
from AdversarialTensors.my_progress_bar import MyProgressBar
from AdversarialTensors.LightningModel import LightningModel
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
#from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import os

if __name__ == '__main__':
    # create parser for command-line arguments
    parser = argparse.ArgumentParser(description='Model Trainer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--init_lr', metavar='init_lr',
        type=float,
        default=1e-1,
        help='init_lr')

    parser.add_argument('--batch_size', metavar='batch_size',
        type=int,
        default=128,
        help='batch_size')

    parser.add_argument('--model_name', metavar='model_name',
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50', "wresnet28_10"],
        help='model name')

    parser.add_argument('--max_epoch', metavar='max_epoch',
        type=int,
        default=200,
        help='max_epoch')

    parser.add_argument('--dataset_name', metavar='dataset_name',
        default='cifar10',
        choices=['cifar10', 'cifar100', 'mnist', "fmnist", "imagenet"],
        help='model name')

    parser.add_argument('--nfolds', metavar='nfolds',
        type=int,
        default=8,
        help='nfolds')

    parser.add_argument('--fold_ind', metavar='fold_ind',
        type=int,
        default=0,
        help='fold_ind')

    parser.add_argument('--rm_old_model', metavar='rm_old_model',
        type=bool,
        default=True,
        help='Remove the old model (if exists)')

    parser.add_argument('--device_id', metavar='device_id',
        type=int,
        default=0,
        help='device_id')

    # Arguments initialization
    # ...
    args = parser.parse_args()
    print(args)

    # Device and basic parameters setup
    device = f"cuda:{args.device_id}"
    accelerator = 'gpu'
    if not torch.cuda.is_available():
        import os
        num_cores = os.cpu_count()
        device = 'cpu'
        accelerator = device
        args.device_id = num_cores

    batch_size = args.batch_size
    lr = args.init_lr
    model_name = args.model_name
    nfolds = args.nfolds
    dataset_name = args.dataset_name
    fold_ind = args.fold_ind
    max_epoch = args.max_epoch
    # Set random seed for reproducibility
    pl.seed_everything(fold_ind, workers=True)

    # Load dataset and prepare DataLoader
    data_params={'batch_size':batch_size,'num_workers':4,'shuffle':True,'normalize':True, 'nfolds':nfolds}
    dataloader = DatasetLoader(name=dataset_name,params=data_params)
    splits, testloader, classes, mean, std = dataloader.fit()
    if nfolds==1:
        my_training_data = splits
        my_valid_data = None
    else:
        my_training_data = splits[fold_ind][0]
        my_valid_data = splits[fold_ind][1]
    # Initialize the model
    model = LightningModel(model_name=model_name, lr=lr, 
                           batch_size=batch_size, 
                           num_batches=len(my_training_data),
                           num_classes=len(classes))

    # Setup logger
    logger_folder = f"log_{dataset_name}_{model_name}"
    log_sub_folder = f"{fold_ind}_{nfolds}"
    logger = CSVLogger(logger_folder, name=log_sub_folder)
    '''
    logger = WandbLogger(project=f"{logger_folder}", # group runs
                         name=f"{log_sub_folder}",
                         log_model='all') # log all new checkpoints during training
    '''    
    # Initialize a trainer
    # set refresh_rate to 0 to disable
    bar = MyProgressBar(refresh_rate=0)
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       min_delta=0.001,
       patience=9999,
       verbose=False,
       mode='min'
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_acc",
                                          mode="max",
                                          save_top_k=1,
                                          dirpath=f"my_checkpoints/{dataset_name}/",
                                          filename=f"{model_name}_{fold_ind}_{nfolds}")
    # before running the trainer cleanup the existing models
    if args.rm_old_model:
        old_model_fpath = f"my_checkpoints/{dataset_name}/{model_name}_{fold_ind}_{nfolds}.ckpt"
        if os.path.exists(old_model_fpath):
            os.remove(old_model_fpath)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=args.device_id,
        max_epochs=max_epoch,
        logger=logger,
        callbacks=[bar, early_stop_callback, checkpoint_callback],
    )
    trainer.fit(model, my_training_data, my_valid_data)

    best_model_path = trainer.checkpoint_callback.best_model_path


