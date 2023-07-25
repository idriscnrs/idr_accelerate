# idr_accelerate

## Description

Make Accelerate on Jean Zay easy.

## Usage
idr_accelerate is a script to create the right config files to use Accelerate with several nodes on Jean Zay. It is a launcher which should be run on every node (with srun for instance) and given the training script. It is possible to use a config file with it for accelerate, as well as give it flags for the accelerate launcher or your own script.


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
module load llm

srun idr_accelerate --config_file myconfig.json --zero_stage 3 --epochs 100 train.py --lr 0.5
```

## Installation

```bash
git clone https://idrforge.prive.idris.fr/assistance/installations/idr_accelerate.git
cd idr_accelerate
pip install .
```
