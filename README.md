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


## Accelerate launch arguments
```
Accelerate launch command [-h] 
[--config_file CONFIG_FILE] [--quiet] [--cpu] [--multi_gpu] [--tpu] [--ipex]
[--mixed_precision {no,fp16,bf16,fp8}] [--num_processes NUM_PROCESSES] [--num_machines NUM_MACHINES]
[--num_cpu_threads_per_process NUM_CPU_THREADS_PER_PROCESS]
[--dynamo_backend {no,eager,aot_eager,inductor,nvfuser,aot_nvfuser,aot_cudagraphs,ofi,fx2trt,onnxrt,ipex}]
[--dynamo_mode {default,reduce-overhead,max-autotune}] [--dynamo_use_fullgraph] [--dynamo_use_dynamic]
[--use_deepspeed] [--use_fsdp] [--use_megatron_lm] [--use_xpu] [--gpu_ids GPU_IDS] [--same_network]
[--machine_rank MACHINE_RANK] [--main_process_ip MAIN_PROCESS_IP] [--main_process_port MAIN_PROCESS_PORT]
[-t TEE] [--role ROLE] [--rdzv_backend RDZV_BACKEND] [--rdzv_conf RDZV_CONF] [--max_restarts MAX_RESTARTS]
[--monitor_interval MONITOR_INTERVAL] [-m] [--no_python] [--tpu_cluster] [--no_tpu_cluster]
[--tpu_use_sudo] [--vm VM] [--env ENV] [--main_training_function MAIN_TRAINING_FUNCTION] [--downcast_bf16]
[--deepspeed_config_file DEEPSPEED_CONFIG_FILE] [--zero_stage ZERO_STAGE]
[--offload_optimizer_device OFFLOAD_OPTIMIZER_DEVICE] [--offload_param_device OFFLOAD_PARAM_DEVICE]
[--offload_optimizer_nvme_path OFFLOAD_OPTIMIZER_NVME_PATH]
[--offload_param_nvme_path OFFLOAD_PARAM_NVME_PATH]
[--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--gradient_clipping GRADIENT_CLIPPING]
[--zero3_init_flag ZERO3_INIT_FLAG] [--zero3_save_16bit_model ZERO3_SAVE_16BIT_MODEL]
[--deepspeed_hostfile DEEPSPEED_HOSTFILE] [--deepspeed_exclusion_filter DEEPSPEED_EXCLUSION_FILTER]
[--deepspeed_inclusion_filter DEEPSPEED_INCLUSION_FILTER]
[--deepspeed_multinode_launcher DEEPSPEED_MULTINODE_LAUNCHER] [--fsdp_offload_params FSDP_OFFLOAD_PARAMS]
[--fsdp_min_num_params FSDP_MIN_NUM_PARAMS] [--fsdp_sharding_strategy FSDP_SHARDING_STRATEGY]
[--fsdp_auto_wrap_policy FSDP_AUTO_WRAP_POLICY]
[--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP]
[--fsdp_backward_prefetch_policy FSDP_BACKWARD_PREFETCH_POLICY]
[--fsdp_state_dict_type FSDP_STATE_DICT_TYPE] [--fsdp_forward_prefetch FSDP_FORWARD_PREFETCH]
[--fsdp_use_orig_params FSDP_USE_ORIG_PARAMS] [--fsdp_sync_module_states FSDP_SYNC_MODULE_STATES]
[--megatron_lm_tp_degree MEGATRON_LM_TP_DEGREE] [--megatron_lm_pp_degree MEGATRON_LM_PP_DEGREE]
[--megatron_lm_num_micro_batches MEGATRON_LM_NUM_MICRO_BATCHES]
[--megatron_lm_sequence_parallelism MEGATRON_LM_SEQUENCE_PARALLELISM]
[--megatron_lm_recompute_activations MEGATRON_LM_RECOMPUTE_ACTIVATIONS]
[--megatron_lm_use_distributed_optimizer MEGATRON_LM_USE_DISTRIBUTED_OPTIMIZER]
[--megatron_lm_gradient_clipping MEGATRON_LM_GRADIENT_CLIPPING] [--aws_access_key_id AWS_ACCESS_KEY_ID]
[--aws_secret_access_key AWS_SECRET_ACCESS_KEY] [--debug]
    training_script ...
```




## Local installation for dev

```bash
module load llm
mkdir .local_accelerate
export PYTHONUSERBASE=$PWD/.local__accelerate:$PYTHONUSERBASE

#mettre dans le slurm
export PATH=$PWD/.local__accelerate/bin:$PATH

```