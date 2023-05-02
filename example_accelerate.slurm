#!/bin/bash
#SBATCH --job-name=example_accelerate
#SBATCH --output=example_accelerate.out
#SBATCH --error=example_accelerate.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=40
#SBATCH --account=<add account>@v100
#SBATCH -C v100-16g


## load Pytorch module
module purge

module load bloom

## launch script on every node
set -x

# code execution
python idr_accelerate.py
srun bash -c 'accelerate launch --config ./config_accelerate_rank${SLURM_PROCID}.yaml finetune_imdb_accelerate.py'