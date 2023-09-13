init_lr="1e-2"
batch_size="256"
model_name="resnet18" #wresnet28_10
max_epoch="200"
dataset_name="cifar10"
nfolds=8
fold_start=0 #place holder
fold_end=4 #place holder
num_gpus=4

export init_lr
export batch_size
export model_name
export max_epoch
export dataset_name
export nfolds
export fold_start
export fold_end
export num_gpus


i=0
for((i=0;i<$((nfolds/num_gpus));++i)) do
    fold_start=$((i*num_gpus))
    fold_end=$((fold_start+num_gpus))
    #bash  train_job.sh
    sbatch train_job.sh
done
