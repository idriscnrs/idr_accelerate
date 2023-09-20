#!/bin/bash
#SBATCH --job-name=example_deepspeed-ZeRO
#SBATCH --output=example_deepspeed-ZeRO.out
#SBATCH --error=example_deepspeed-ZeRO.out

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
srun idr_accelerate --zero_stage 2 --mixed_precision fp16 train.py --lr 0.5


