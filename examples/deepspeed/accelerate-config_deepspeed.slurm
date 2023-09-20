#!/bin/bash
#SBATCH --job-name=accelerate-config_deepspeed
#SBATCH --output=slurm_log/accelerate-config_deepspeed.out
#SBATCH --error=slurm_log/accelerate-config_deepspeed.out

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH -C v100-16g
#SBATCH --hint=nomultithread

#SBATCH --account=sos@v100
#SBATCH --time=00:30:00
#SBATCH --qos=qos_gpu-dev

## Loading Jean-Zay module
module purge
module load pytorch-gpu/py3/2.0.1

# Code execution
# We give accelerate an accelerate configuration file 
srun idr_accelerate --config_file configs/accelerate_deepspeed.yaml train.py --lr 0.5
