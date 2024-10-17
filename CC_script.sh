#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --account=def-mtaylor3
#SBATCH --output=/home/shang8/scratch/slurm_out/%A.out
#SBATCH --mail-user=shang8@ualberta.ca
#SBATCH --mail-type=ALL


module load python/3.10
module load cuda
source /home/shang8/scratch/FPGA_env/bin/activate
wandb offline

python3 main.py --wandb --amp --group_name C30B --seed 0 \
                --num_rollout_workers 8 --num_cpus_per_worker 4 --num_gpus_per_worker 0.5 \
                --min_num_episodes_per_worker 20 --num_target_blocks 30 \
                
