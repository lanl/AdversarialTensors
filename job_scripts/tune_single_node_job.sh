#!/bin/bash --login
#SBATCH --time=3:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=16        # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=128G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --gpus=a100:4 # (request 2 k80 GPUs for entire job)
#SBATCH --job-name train      # you can give your job a name for easier identification (same as -J)
###SBATCH --constraint=nif
########## Command Lines for Job Running ##########

conda activate base

python parameter_search.py --model_name ${model_name} \
                           --dataset_name ${dataset_name} \
                           --attack_type ${attack_type} \
                           --batch_size ${batch_size} \
                           --nfolds ${nfolds} \
                           --fold_ind ${fold_ind} \
                           --num_samples ${num_samples}

