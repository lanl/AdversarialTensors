#!/bin/bash --login
#SBATCH --time=00:19:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=2                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=2
###SBATCH --exclusive
#SBATCH --cpus-per-task=16        # number of CPUs (or cores) per task (same as -c)
#SBATCH --tasks-per-node=1
#SBATCH --mem=128G            # memory required
#SBATCH --job-name train      # you can give your job a name for easier identification (same as -J)
#SBATCH --constraint=nif
#SBATCH --gpus-per-task=a100:2
########## Command Lines for Job Running ##########

conda activate base
CPUS_PER_TASK=16
GPUS_PER_TASK=4

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${CPUS_PER_TASK}" --num-gpus "${GPUS_PER_TASK}" --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${CPUS_PER_TASK}" --num-gpus "${GPUS_PER_TASK}" --block &
    sleep 5
done

python parameter_search.py --model_name ${model_name} \
                           --dataset_name ${dataset_name} \
                           --attack_type ${attack_type} \
                           --batch_size ${batch_size} \
                           --nfolds ${nfolds} \
                           --fold_ind ${fold_ind} \
                           --num_samples ${num_samples}
