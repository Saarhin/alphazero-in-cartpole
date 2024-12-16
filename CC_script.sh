#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/saarhin/scratch/chip/slurm_out/%A.out
#SBATCH --mail-user=samini1@ualberta.ca
#SBATCH --mail-type=ALL

echo $1 # c_init
echo $2 # num_simulations

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline # log offline
export VTR_ROOT=/home/shang8/scratch/vtr-verilog-to-routing
export results=$SLURM_TMPDIR/results
cp -R /home/saarhin/scratch/chip/alphazero-in-cartpole/data $SLURM_TMPDIR/data
export data=$SLURM_TMPDIR/data
export HEAD_NODE=$(hostname)
export RAY_PORT=34567

module load python/3.10
module load cuda
source /home/saarhin/scratch/chip/chip_env/bin/activate
wandb offline

ray start --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT --num-cpus=32 --num-gpus=2 --block &
sleep 20

# c56b
# PYTHONUNBUFFERED=1 python3 -u main.py --wandb --amp --cc --group_name c56b_longer --seed 0 \
#                 --num_rollout_workers 8 --num_cpus_per_worker 4 --num_envs_per_worker 20 --num_gpus_per_worker 0.25 \
#                 --min_num_episodes_per_worker 20 --num_target_blocks 56 --num_simulations $2 \
#                 --training_steps 70 --c_init $1 

# c30b
# PYTHONUNBUFFERED=1 python3 -u main.py --wandb --amp --cc --group_name c30b --seed 0 \
#                 --num_rollout_workers 8 --num_cpus_per_worker 4 --num_envs_per_worker 10 --num_gpus_per_worker 0.25 \
#                 --min_num_episodes_per_worker 20 --num_target_blocks 30 --num_simulations $2 \
#                 --training_steps 35 --c_init $1 

# # c15b
PYTHONUNBUFFERED=1 python3 -u main.py --wandb --amp --cc --group_name c15b --seed 0 \
                --num_rollout_workers 8 --num_cpus_per_worker 4 --num_envs_per_worker 10 --num_gpus_per_worker 0.25 \
                --min_num_episodes_per_worker 20 --num_target_blocks 15 --num_simulations $2 \
                --training_steps 25 --c_init $1 

cp -r $results/* /home/saarhin/scratch/chip/alphazero-in-cartpole/results/

# # c5b
# PYTHONUNBUFFERED=1 python3 -u main.py --wandb --amp --cc --group_name c15b --seed 0 \
#                 --num_rollout_workers 8 --num_cpus_per_worker 4 --num_gpus_per_worker 0.25 \
#                 --min_num_episodes_per_worker 20 --num_target_blocks 15 --num_simulations 100 \
#                 --training_steps 25 --c_init 1.25
                
