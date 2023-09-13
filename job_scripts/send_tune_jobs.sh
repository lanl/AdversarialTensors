model_name="resnet18" #wresnet28_10
dataset_name="cifar10"
attack_type="Linf"
batch_size="256"
nfolds=8
fold_ind=0 #place holder
num_samples=200

export model_name
export dataset_name
export attack_type
export batch_size
export nfolds
export fold_ind
export num_samples

i=0
for((i=0;i<$nfolds;++i)) do
    fold_ind=$i
    sbatch tune_single_node_job.sh
done
