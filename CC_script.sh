#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/shang8/scratch/slurm_out/%A.out
#SBATCH --mail-user=shang8@ualberta.ca
#SBATCH --mail-type=ALL

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline # log offline
export VTR_ROOT=/home/shang8/scratch/vtr-verilog-to-routing
export results=$SLURM_TMPDIR/results
cp -R /home/shang8/scratch/alphazero-in-cartpole/data $SLURM_TMPDIR/data
export data=$SLURM_TMPDIR/data
export HEAD_NODE=$(hostname)
export RAY_PORT=34567

module load python/3.10
module load cuda
source /home/shang8/scratch/MCTS_env/bin/activate
wandb offline

ray start --head --node-ip-address=$HEAD_NODE --port=$RAY_PORT --num-cpus=32 --num-gpus=2 --block &
sleep 20

PYTHONUNBUFFERED=1 python3 -u main.py --wandb --amp --cc --group_name c5b --seed 0 \
                --num_rollout_workers 8 --num_cpus_per_worker 4 --num_gpus_per_worker 0.25 \
                --min_num_episodes_per_worker 20 --num_target_blocks 5 --num_simulations 50 \
                
