#!/bin/bash
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=10G
#SBATCH --time=20:00:00
#SBATCH --account=def-mtaylor3
#SBATCH --output=/home/shang8/scratch/slurm_out/%A.out
#SBATCH --mail-user=shang8@ualberta.ca
#SBATCH --mail-type=ALL


module load python/3.10
module load cuda
source /home/shang8/scratch/FPGA_env/bin/activate
wandb offline

python3 main.py --wandb --amp --group_name C30B --seed 0 \
                --num_rollout_workers 8 --num_cpus_per_worker 8 --num_gpus_per_worker 1