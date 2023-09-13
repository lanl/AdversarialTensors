#@author: Mehmet Cagri

import torch
from AdversarialTensors.datasets import DatasetLoader
from AdversarialTensors.denoiser import Denoiser
from AdversarialTensors.simple_attack import Attacks
from AdversarialTensors.utils import eval_accuracy_dataloader, eval_accuracy_w_reconst_dataloader
import numpy as np
import pickle
from AdversarialTensors.normalize import Normalize
import optuna
import argparse
from AdversarialTensors.LightningModel import LightningModel
from pathlib import Path
from AdversarialTensors.model import FinalModel
import pandas as pd

# Function to process data and group it into batches
def process_data(data, batch_size=64):
    new_data = []
    for X,Y in data:
        lenn = len(Y)
        for bs in range(0, lenn, batch_size):
            n_x = X[bs:bs+batch_size]
            n_y = Y[bs:bs+batch_size]
            if type(n_x) == np.ndarray:
                n_x = torch.from_numpy(n_x)
                n_y = torch.from_numpy(n_y)
            new_data.append((n_x, n_y))
    return new_data

if __name__ == '__main__':
    # create parser for command-line arguments
    parser = argparse.ArgumentParser(description='Final Evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_name', metavar='model_name',
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50', "wresnet28_10"],
        help='model name')

    parser.add_argument('--dataset_name', metavar='dataset_name',
        default='cifar10',
        choices=['cifar10', 'cifar100', 'mnist', "fmnist", "imagenet"],
        help='model name')

    parser.add_argument('--attack_type', metavar='attack_type',
        default='Linf',
        choices=['L2', 'Linf'],
        help='attack_type')

    parser.add_argument('--batch_size', metavar='batch_size',
        type=int,
        default=128,
        help='batch_size')

    parser.add_argument('--nfolds', metavar='nfolds',
        type=int,
        default=8,
        help='nfolds')

    parser.add_argument('--fold_ind', metavar='fold_ind',
        type=int,
        default=0,
        help='fold_ind')

    parser.add_argument('--eval_top_k', metavar='eval_top_k',
        type=int,
        default=1,
        help='eval_top_k')

    parser.add_argument('--device_id', metavar='device_id',
        type=int,
        default=0,
        help='device_id')

    args = parser.parse_args()
    print(args)

    # Set device for computations
    device = f"cuda:{args.device_id}"

    # Load the pre-trained model
    model_path = f"my_checkpoints/{args.dataset_name}/{args.model_name}_{args.fold_ind}_{args.nfolds}.ckpt"
    base_model = LightningModel.load_from_checkpoint(model_path).model

    # Prepare DataLoader
    data_params = {'batch_size': args.batch_size, 'num_workers': 4, 'shuffle': False, 'normalize': False,
                   'nfolds': args.nfolds}
    dataloader = DatasetLoader(name=args.dataset_name, params=data_params)
    splits, testloader, classes, mean, std = dataloader.fit()

    # Add normalization layer to the model and move it to device
    base_model = torch.nn.Sequential(
        Normalize(mean, std),  # Normalize using the dataset's mean and std
        base_model,
    ).eval().to(device)

    # Load attack data
    log_path = f"attack_log_{args.dataset_name}/{args.model_name}/{args.fold_ind}_{args.nfolds}_{args.attack_type}"
    attack_data_location = f"{log_path}/final_test_attack_data.pkl"
    with open(attack_data_location, "rb") as fptr:
        attacks = pickle.load(fptr)

    # Process attack data
    attacks = process_data(attacks, batch_size=args.batch_size)

    # Read parameter tuning results from CSV
    csv_folder = f"tune_csv_results/parameter_search_{args.dataset_name}/{args.model_name}/{args.fold_ind}_{args.nfolds}_{args.attack_type}"
    csv_path = f"{csv_folder}/results.csv"
    results = pd.read_csv(csv_path)

    # Sort by fitness and select top K configurations for further evaluation
    results = results.sort_values('fitness', ascending=False)
    end_i = min(args.eval_top_k, len(results))
    new_results_df = results.iloc[:end_i].copy()  # Deep copy to avoid modifying original DataFrame

    # Initialize new columns for final results
    new_columns = {"final_clean_acc": [], "final_adv_acc": [], "final_clean_rec_err": [], "final_adv_rec_err": []}

    # Evaluate each configuration and collect results
    for i in range(end_i):
        # Initialize denoiser
        denoiser = Denoiser(method='tt', device=device, tensor_params={'factors':None,'init':'svd',
                                                                      'svd':'truncated_svd',
                                                                      'tol':1e-3,'max_iter':1},
                     verbose=False,
                     patch_params={'patch_size':results['patch_size'][i],
                     'stride':results['stride'][i],'channels':3},
                     data_mode='single',
                     ranks=[1, results['rank2'][i], results['rank3'][i], results['rank4'][i], 1])
        # Initialize and evaluate final model
        model = FinalModel(base_model, denoiser).to(device).eval()
        clean_acc, clean_rec_err = eval_accuracy_w_reconst_dataloader(model, testloader, device)
        adv_acc, adv_rec_err = eval_accuracy_w_reconst_dataloader(model, attacks, device)

        # Append results to new column
        new_columns['final_clean_acc'].append(clean_acc)
        new_columns['final_clean_rec_err'].append(clean_rec_err)
        new_columns['final_adv_acc'].append(adv_acc)
        new_columns['final_adv_rec_err'].append(adv_rec_err)

    # Add new results to DataFrame
    for k,v in new_columns.items():
        new_results_df[k] = v

    # Save the final results to CSV
    new_results_df.to_csv(f"{csv_folder}/final_results.csv", index=False)



