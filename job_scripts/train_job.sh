##!/bin/bash --login
##SBATCH --time=3:00:00
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=4
##SBATCH --cpus-per-task=4
##SBATCH --mem=128G
##SBATCH --gpus=a100:4
##SBATCH --job-name train

########## Command Lines for Job Running ##########

#conda activate base
cd ../examples/
i=0
for((i=$fold_start;i<$fold_end;++i)) do
    device_id=$((i%num_gpus))
    python train_model.py --init_lr  ${init_lr} --batch_size ${batch_size} \
                          --model_name ${model_name} \
                          --max_epoch ${max_epoch} --dataset_name ${dataset_name} \
                          --nfolds ${nfolds} --fold_ind ${i} --device_id ${device_id} &
done

# wait for processes to finish
wait

