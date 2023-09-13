model_name="resnet18" #wresnet28_10
dataset_name="cifar10"
nfolds=8
attack_type="Linf" #or "L2"
cd ../examples/
i=0
for((i=0;i<$nfolds;++i)) do
    python collect_tune_results.py --model_name ${model_name} \
                                   --dataset_name ${dataset_name} \
                                   --fold_ind ${i} \
                                   --nfolds ${nfolds} \
                                   --attack_type ${attack_type}
done
