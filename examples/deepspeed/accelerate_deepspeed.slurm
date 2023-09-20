#!/bin/bash
#SBATCH --job-name=accelerate_deepspeed
#SBATCH --output=slurm_log/accelerate_deepspeed.out
#SBATCH --error=slurm_log/accelerate_deepspeed.out

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
module load llm

# Code execution
# We give accelerate CLI args to use deepspeed with ZeRO stage 2
srun idr_accelerate --use_deepspeed --zero_stage 2 --mixed_precision fp16 train.py --lr 0.5
