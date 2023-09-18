#!/bin/bash
#SBATCH --job-name=accelerate-config_deepspeed-config
#SBATCH --output=slurm_log/accelerate-config_deepspeed-config.out
#SBATCH --error=slurm_log/accelerate-config_deepspeed-config.out

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

export PYTHONPATH=/gpfsdswork/projects/idris/sos/ssos029/idr_accelerate/.local_accelerate/lib/python3.10/site-packages/:$PYTHONPATH


# Code execution
# We give accelerate an accelerate configuration file which in turn points to a 2nd deepspeed configuration file (configs/zero_stage2_config.json)
srun idr_accelerate --config_file=configs/accelerate_deepspeed-config.yaml train.py --lr 0.5