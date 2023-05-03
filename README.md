# idr_accelerate
## WORK IN PROGRESS !!!


## Description

Make Accelerate on Jean Zay easy.

## Usage
idr_accelerate.py is a script to create the right config files to use Accelerate with several nodes on Jean Zay.
It should be run once on the main process node first to create the config files, then eache node will run (thanks to srun) the python script with the right config file (with the command srun bash -c '...').

Follow the example to have a better understanding.

```bash
#!/bin/bash
#SBATCH --job-name=example_accelerate
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --account=account@v100

## load module
module purge
module load bloom

## echo launch commands
set -x

# code execution
idr_accelerate
srun bash -c 'accelerate launch --config ./config_accelerate_rank${SLURM_PROCID}.yaml train.py'
```

## Installation

```bash
git clone https://idrforge.prive.idris.fr/assistance/installations/idr_accelerate.git
cd idr_accelerate
cp idr_accelerate   $CONDA_PREFIX/bin/.
```

## Local test
```bash
chmod +x idr_accelerate
# add idr_accelerate dir to the path
export $PATH=$PWD:PATH

# use from anywhere
idr_accelerate
```
