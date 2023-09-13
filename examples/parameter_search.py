#@author: Mehmet Cagri

import ray
ray.init()
import torch
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
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

# ray handles the specific device assignment
device = 'cuda'

def process_data(data, batch_size=64):
    """
    Processes data by splitting into smaller batches.

    Parameters
    ----------
    data : list of tuples
        List of tuples, each containing a batch of images and labels.
    batch_size : int, optional
        The size of each batch. Default is 64.

    Returns
    -------
    new_data : list of tuples
        List of new data batches.
    """
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


def calculate_min_dims(all_dims):
    """
    Calculate minimum dimensions for tensor decomposition.

    Parameters
    ----------
    all_dims : tuple
        Dimensions of the tensor.

    Returns
    -------
    all_min_dims : list
        List of minimum dimensions.
    """
    all_min_dims = []
    for i in range(len(all_dims)):
        dim1 = all_dims[i]
        dim2 = np.prod(all_dims[:i] + all_dims[i+1:])
        min_dim = min(dim1,dim2)
        all_min_dims.append(min_dim)
    return all_min_dims

def create_powers2(start,end):
    """
    Create a list of powers of 2 between start and end.

    Parameters
    ----------
    start : int
        Starting value.
    end : int
        Ending value.

    Returns
    -------
    res : list
        List of powers of 2.
    """
    import math
    first_power = math.floor(math.log(start, 2))
    last_power = math.floor(math.log(end, 2))
    res = []
    for i in range(first_power, last_power + 1):
        res.append(2**i)
    if end > 2**last_power:
        res.append(end)
    return res

def define_by_run_func(trial):
    """
    Define the search space for Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.

    Returns
    -------
    None or dict
        Returns a dict with constant values or None.
    """
    global image_size
    global channels
    padding = 1
    patch_size = trial.suggest_categorical("patch_size", [4, 8, 16, 24])
    stride = trial.suggest_categorical("stride", [1, 2, 4])
    num_patches = ((image_size + 2 * padding - patch_size) // stride) + 1
    num_patches *= num_patches
    all_dims = (num_patches, channels, patch_size, patch_size)
    all_min_dims = calculate_min_dims(all_dims)
    # Define-by-run allows for conditional search spaces.
    #trial.suggest_categorical("rank2", [channels,])
    rank34_max = max([all_min_dims[2], 4])
    #trial.suggest_int("rank34", 4, rank34_max, 4)
    #trial.suggest_categorical("rank34", [rank34_max,])
    trial.suggest_int("rank34", patch_size, patch_size, 1)
    rank1_min = min([12, all_min_dims[0]//3 + 1])
    rank1_max = min([96, all_min_dims[0]//2 + 1])
    trial.suggest_int("rank1", rank1_min, rank1_max, 4)

# 1. Wrap a PyTorch model in an objective function.
def objective(config, clean_data, adv_data, base_model):
    """
    Objective function to evaluate the model performance.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    clean_data : list of tuples
        List of clean data batches.
    adv_data : list of tuples
        List of adversarial data batches.
    base_model : PyTorch model
        Base model to be evaluated.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    denoiser = Denoiser(method='tucker', device=0, tensor_params={'factors':None,'init':'svd',
                                                                  'svd':'truncated_svd',
                                                                  'tol':1e-3,'max_iter':1},
                 verbose=False,
                 patch_params={'patch_size':config['patch_size'], 'stride':config['stride'],'channels':3},
                 data_mode='single',
                 ranks=[config['rank1'], 3, config['rank34'],config['rank34']])

    model = FinalModel(base_model, denoiser).to(device).eval()

    #attacker = Attacks(model=base_model, attack = 'fgsm', attack_params = {'norm': 2, 'eps': 8 / 255, 'version': 'custom', 'log_dir': 'autoattack/',
    #                                        'seed': 99, 'exp': 'all'}, device = 0)

    clean_acc, clean_rec_err = eval_accuracy_w_reconst_dataloader(model, clean_data, device)
    adv_acc, adv_rec_err = eval_accuracy_w_reconst_dataloader(model, adv_data, device)

    alpha = 0.5
    fitness_val = adv_acc * alpha + clean_acc * (1 - alpha) 
    return {"fitness": fitness_val,
            "clean_acc":clean_acc,
            "adv_acc":adv_acc,
            "clean_rec_err":clean_rec_err,
            "adv_rec_err":adv_rec_err}


def define_by_run_func_tt(trial):
    """
    Define the search space for Optuna trial for tensor train decomposition.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.

    Returns
    -------
    None or dict
        Returns a dict with constant values or None.
    """
    global image_size
    global channels
    
    padding = 1
    patch_size = trial.suggest_categorical("patch_size", [4, 8, 16, 24])
    stride = trial.suggest_categorical("stride", [1, 2, 4])
    num_patches = ((image_size + 2 * padding - patch_size) // stride) + 1
    num_patches *= num_patches
    all_dims = (num_patches, channels, patch_size, patch_size)
    all_min_dims = calculate_min_dims(all_dims)
    r2_min, r2_max = 12, int(min(num_patches, channels * patch_size**2) * 2/3)
    if r2_min > r2_max:
        r2_min = r2_max
    r2 = trial.suggest_int("rank2", r2_min, r2_max, 4)
    #r3_min, r3_max = 8, min(r2 * channels, patch_size**2)
    r3_min, r3_max = 8, min(r2 * patch_size, patch_size*channels)
    if r3_min > r3_max:
        r3_min = r3_max
    r3 = trial.suggest_int("rank3", r3_min, r3_max, 4)
    #r4_min, r4_max = 4, patch_size
    r4_min, r4_max = 3, 3
    if r4_min > r4_max:
        r4_min = r4_max
    r4 = trial.suggest_int("rank4", r4_min, r4_max, 4)

# 1. Wrap a PyTorch model in an objective function.
def objective_tt(config, clean_data, adv_data, base_model):
    """
    Objective function to evaluate the model performance for tensor train decomposition.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    clean_data : list of tuples
        List of clean data batches.
    adv_data : list of tuples
        List of adversarial data batches.
    base_model : PyTorch model
        Base model to be evaluated.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    denoiser = Denoiser(method='tt', device=0, tensor_params={'factors':None,'init':'svd',
                                                                  'svd':'truncated_svd',
                                                                  'tol':1e-3,'max_iter':1},
                 verbose=False,
                 patch_params={'patch_size':config['patch_size'], 'stride':config['stride'],'channels':3},
                 data_mode='single',
                 ranks=[1, config['rank2'], config['rank3'], config['rank4'], 1])

    model = FinalModel(base_model, denoiser).to(device).eval()

    #attacker = Attacks(model=base_model, attack = 'fgsm', attack_params = {'norm': 2, 'eps': 8 / 255, 'version': 'custom', 'log_dir': 'autoattack/',
    #                                        'seed': 99, 'exp': 'all'}, device = 0)

    clean_acc, clean_rec_err = eval_accuracy_w_reconst_dataloader(model, clean_data, device)
    adv_acc, adv_rec_err = eval_accuracy_w_reconst_dataloader(model, adv_data, device)

    alpha = 0.5
    fitness_val = adv_acc * alpha + clean_acc * (1 - alpha) 
    return {"fitness": fitness_val,
            "clean_acc":clean_acc,
            "adv_acc":adv_acc,
            "clean_rec_err":clean_rec_err,
            "adv_rec_err":adv_rec_err}
           
if __name__ == '__main__':
    # create parser for command-line arguments
    parser = argparse.ArgumentParser(description='Parameter Search',
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
          
    parser.add_argument('--num_samples', metavar='num_samples',
        type=int,
        default=200,
        help='num_samples for the parameter search') 
        
    args = parser.parse_args()
    print(args)
    
    model_path = f"my_checkpoints/{args.dataset_name}/{args.model_name}_{args.fold_ind}_{args.nfolds}.ckpt"
    
    base_model = LightningModel.load_from_checkpoint(model_path).model

    data_params={'batch_size':args.batch_size,'num_workers':4,'shuffle':False,'normalize':False, 'nfolds':args.nfolds}
    dataloader = DatasetLoader(name=args.dataset_name,params=data_params)
    splits, testloader, classes, mean, std = dataloader.fit()

    my_training_data = splits[args.fold_ind][0]
    my_valid_data = splits[args.fold_ind][1]
    
    base_model = torch.nn.Sequential(
        Normalize(mean, std), #from dataloader
        base_model,
    ).eval().to(device)
    
    log_path = f"attack_log_{args.dataset_name}/{args.model_name}/{args.fold_ind}_{args.nfolds}_{args.attack_type}"
    attack_data_location = f"{log_path}/attack_data.pkl"
    with open(attack_data_location, "rb") as fptr:
        attacks = pickle.load(fptr)

    attacks = process_data(attacks, batch_size=args.batch_size)
    my_valid_data = process_data(my_valid_data, batch_size=args.batch_size)
    
    b_s, channels, w, h = my_valid_data[0][0].shape
    image_size = w # assuming image is a rectangle
    
    algo = OptunaSearch(space=define_by_run_func_tt, metric=['clean_acc', 'adv_acc'], mode=['max','max'], sampler=optuna.samplers.RandomSampler())

    trainable_with_resources = tune.with_resources(objective_tt, {"gpu": 1, 'cpu':4})
    final_objective = tune.with_parameters(trainable_with_resources, clean_data=my_valid_data, adv_data=attacks, base_model=base_model)
    # 3. Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
    par_search_path = f"parameter_search_{args.dataset_name}/{args.model_name}/{args.fold_ind}_{args.nfolds}_{args.attack_type}"
    par_search_dir = Path(par_search_path)
    par_search_dir.mkdir(parents=True, exist_ok=True)
    tuner = tune.Tuner(
        final_objective,
        tune_config=tune.TuneConfig(
            metric='fitness',
            mode='max',
            search_alg=algo,
            num_samples=args.num_samples,
        ),
        run_config=air.RunConfig(local_dir=par_search_dir, name="search_results",
                                sync_config=tune.SyncConfig(syncer=None)  # Disable syncing
        )
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
