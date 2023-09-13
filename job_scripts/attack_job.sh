#!/bin/bash --login
#SBATCH --time=3:00:00             
#SBATCH --nodes=1                 
#SBATCH --ntasks=4                  
#SBATCH --cpus-per-task=4        
#SBATCH --mem=128G           
#SBATCH --gpus=a100:4 
#SBATCH --job-name train    
####SBATCH --constraint=nif
########## Command Lines for Job Running ##########

conda activate base
cd ../examples/
i=0
for((i=$fold_start;i<$fold_end;++i)) do
    device_id=$((i%num_gpus))
    python generate_attacks_to_save.py --model_name ${model_name} \
                                       --dataset_name ${dataset_name} \
                                       --attack_type ${attack_type} \
                                       --attack_eps ${attack_eps} \
                                       --batch_size ${batch_size} \
                                       --nfolds ${nfolds} \
                                       --fold_ind ${i} \
                                       --device_id ${device_id} & # run it in the background 
done

# wait for processes to finish
wait
