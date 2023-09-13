#@author: Mehmet Cagri
import os
import json
import csv
import argparse
from pathlib import Path

# Main script entry point
if __name__ == '__main__':
    # Create argument parser for parsing command-line arguments
    parser = argparse.ArgumentParser(description='Collect results',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add various command-line options
    parser.add_argument('--model_name', metavar='model_name',
                        default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', "wresnet28_10"],
                        help='model name')

    parser.add_argument('--dataset_name', metavar='dataset_name',
                        default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', "fmnist", "imagenet"],
                        help='dataset name')

    parser.add_argument('--nfolds', metavar='nfolds',
                        type=int,
                        default=8,
                        help='number of folds for cross-validation')

    parser.add_argument('--fold_ind', metavar='fold_ind',
                        type=int,
                        default=0,
                        help='index of the current fold')

    parser.add_argument('--attack_type', metavar='attack_type',
                        default='Linf',
                        choices=['L2', 'Linf'],
                        help='type of adversarial attack to use')

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    # Build the path for storing parameter search data
    par_search_path = f"parameter_search_{args.dataset_name}/{args.model_name}/{args.fold_ind}_{args.nfolds}_{args.attack_type}"
    sub_path = "search_results"
    full_path = f"{par_search_path}/{sub_path}"

    # Initialize result collections
    results_set = set()
    final_res = []

    # Walk through the directory to collect results
    for subdir, dirs, files in os.walk(full_path):
        for dir in dirs:
            d_path = os.path.join(subdir, dir)
            res_file = f"{d_path}/result.json"
            data = None
            with open(res_file) as fptr:
                try:
                    # Load JSON and extract relevant fields
                    data = json.load(fptr)
                    # Various metrics and configurations
                    fitness = data['fitness']
                    clean_acc = data['clean_acc']
                    adv_acc = data['adv_acc']
                    clean_rec_err = data['clean_rec_err']
                    adv_rec_err = data['adv_rec_err']
                    config = data['config']
                    patch_size = config['patch_size']
                    if 'stride' in config:
                        stride = config['stride']
                    else:
                        stride = 1
                    rank1 = 1
                    rank2 = config['rank2']
                    rank3 = config['rank3']
                    rank4 = config['rank4']
                    rank5 = 1
                    time = data['time_total_s']

                    # Create a unique key to avoid duplicates
                    keyy = (patch_size, stride, rank2, rank3, rank4)
                    if keyy not in results_set:
                        results_set.add(keyy)
                        # Append the result to the final result list
                        final_res.append((time, patch_size, stride, rank2, rank3, rank4, clean_acc, adv_acc,
                                          clean_rec_err, adv_rec_err, fitness))
                        print(final_res[-1])
                except:
                    pass

    print(len(final_res))

    # Create directory for saving final results in CSV
    final_path = f"tune_csv_results/{par_search_path}"
    final_path_dir = Path(final_path)
    final_path_dir.mkdir(parents=True, exist_ok=True)

    # Write final results to CSV
    with open(f'{final_path}/results.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(
            ['time', 'patch_size', 'stride', 'rank2', 'rank3', 'rank4', 'clean_acc', 'adv_acc', 'clean_rec_err',
             'adv_rec_err', 'fitness'])
        csv_out.writerows(final_res)
