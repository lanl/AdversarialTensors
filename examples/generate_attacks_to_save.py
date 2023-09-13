#@author: Mehmet Cagri
import torch
from AdversarialTensors.datasets import DatasetLoader
from AdversarialTensors.simple_attack import Attacks
from AdversarialTensors.utils import eval_accuracy_dataloader
import numpy as np
from tqdm import tqdm
import pickle
from AdversarialTensors.normalize import Normalize
from AdversarialTensors.model import FinalModel
import argparse
from AdversarialTensors.LightningModel import LightningModel
from pathlib import Path

if __name__ == '__main__':
    # create parser for command-line arguments
    parser = argparse.ArgumentParser(description='Model Attack Generate',
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
        
    parser.add_argument('--attack_eps', metavar='attack_eps',
        type=str,
        default="8/255",
        help='attack_eps as a string, Ex. 8/255 for Linf, 0.5 for L2') 
        
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
        
    parser.add_argument('--final_test_set', metavar='final_test_set',
        type=bool,
        default=False,
        help='final_test_set')
        
    parser.add_argument('--device_id', metavar='device_id',
        type=int,
        default=0,
        help='device_id')
        
        
    args = parser.parse_args()
    print(args)
    # Set up the computation device (GPU/CPU)
    device = f"cuda:{args.device_id}"

    # Define paths for model checkpoints and attack logs
    model_path = f"my_checkpoints/{args.dataset_name}/{args.model_name}_{args.fold_ind}_{args.nfolds}.ckpt"
    log_path = f"attack_log_{args.dataset_name}/{args.model_name}/{args.fold_ind}_{args.nfolds}_{args.attack_type}"
    attack_log_dir = Path(log_path)
    attack_log_dir.mkdir(parents=True, exist_ok=True)
    # Load the pre-trained model
    base_model = LightningModel.load_from_checkpoint(model_path).model

    # Set dataset loading parameters and initialize the DataLoader
    data_params={'batch_size':args.batch_size,'num_workers':4,'shuffle':False,'normalize':False, 'nfolds':args.nfolds}
    dataloader = DatasetLoader(name=args.dataset_name,params=data_params)
    splits, testloader, classes, mean, std = dataloader.fit()

    my_training_data = splits[args.fold_ind][0]
    my_valid_data = splits[args.fold_ind][1]
    # Choose data based on whether we are running a final test or not
    if args.final_test_set:
        attack_data_location = f"{log_path}/final_test_attack_data.pkl"
        data = testloader
    else:
        attack_data_location = f"{log_path}/attack_data.pkl"
        data = my_valid_data
    # Add normalization layer to the base model
    base_model = torch.nn.Sequential(
        Normalize(mean, std), #from dataloader
        base_model,
    ).eval().to(device)

    # Initialize the attacker
    attacker = Attacks(model=base_model, attack = 'autoattack', attack_params = {'norm': args.attack_type, 'eps': eval(args.attack_eps), 'version': 'custom', 'log_dir': log_path,
                                                'seed': 99, 'exp': 'all'}, device = device)

    all_attack_data = []
    orig_data = []
    i = 0
    # Generate adversarial tests using the attacker
    for X, Y in tqdm(data):
        # Move tensors to device
        X = X.to(device)
        Y = Y.to(device)

        # Generate perturbed data
        X_p = attacker(X, Y).to(device)

        # Convert tensors to NumPy arrays for storage
        X = X.cpu().detach().numpy()
        X_p = X_p.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()

        # Store the attacked and original data
        all_attack_data.append((X_p, Y))
        orig_data.append(X)
        i = i + 1
        #if i == 1:
        #    break
    with open(attack_data_location, 'wb') as handle:
            pickle.dump(all_attack_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

