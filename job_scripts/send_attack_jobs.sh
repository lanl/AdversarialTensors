model_name="resnet18" #wresnet28_10
dataset_name="cifar10"
attack_type="Linf"
attack_eps="8/255"
batch_size="256"
nfolds=8
fold_start=0 #place holder
fold_end=4 #place holder
num_gpus=4

export model_name
export dataset_name
export attack_type
export attack_eps
export batch_size
export nfolds
export fold_start
export fold_end
export num_gpus


i=0
for((i=0;i<$((nfolds/num_gpus));++i)) do
    fold_start=$((i*num_gpus))
    fold_end=$((fold_start+num_gpus))
    sbatch attack_job.sh
done
