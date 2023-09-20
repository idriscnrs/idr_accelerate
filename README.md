# idr_accelerate 🚀
---
## Description

idr_accelerate 🚀 is a python script which allows easy and quick use of [accelerate](https://huggingface.co/docs/accelerate/index) on Jean Zay.
It allows to generate the mandatory configuration scripts for a multi-node use.
It is possible to use a config file with it for accelerate, as well as give it flags for the accelerate launcher or your own script.
idr_accelerate is based on [idr_torch](https://idrforge.prive.idris.fr/assistance/outils/idr_torch)


## Installation
### With [idr-pypi](https://idrforge.prive.idris.fr/assistance/outils/idr_pypi) 🐍 (by default)
```bash
pip install idris[accelerate]
```
### From source
```bash
git clone https://idrforge.prive.idris.fr/assistance/installations/idr_accelerate.git
cd idr_accelerate
pip install .
```

## Get Started
Follow the example to have a better understanding :
```bash
idr_accelerate --help

srun idr_accelerate train.py

srun idr_accelerate train.py {train args}
srun idr_accelerate train.py --lr 0.5 --epochs 100

srun idr_accelerate {config_file} {accelerate args} train.py {train args}
srun idr_accelerate --config_file myconfig.json --zero_stage 3 train.py --lr 0.5

```

Example to run a multi-GPU/node training from slurm script :
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

srun idr_accelerate --config_file myconfig.json --zero_stage 3 train.py --lr 0.5
```

More examples can be found [here !](https://idrforge.prive.idris.fr/assistance/installations/idr_accelerate/examples)

## How it's work

idr_accelerate is a wrapper for accelerate launcher.
It intercepts the arguments/configuration files intended for the accelerate launcher, concatenates/modifies them as necessary for operation on Jean Zay. It then saves an accelerated configuration file specific to the nodes to use (mandatory for multi-node).
It ends up launching the training/inference script with its arguments via the accelerate launcher (which use thr previous accelerate configuration file).

## [Deepspeed in accelerate](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) 

from the official documentation (v0.22.0)
Currently, Accelerate supports following config through the CLI:
```
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them.
`gradient_clipping`: Enable gradient clipping with value.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training. 
```

To be able to tweak more options, you will need to use a DeepSpeed config file.



