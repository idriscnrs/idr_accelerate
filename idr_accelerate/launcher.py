#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
from argparse import REMAINDER, ArgumentParser, _HelpAction
from pathlib import Path
from typing import Any, Dict, Optional

import idr_torch
import torch
import yaml
from accelerate.commands.launch import launch_command, launch_command_parser
from packaging.version import Version


def make_config_file_parser(accelerate_parser: ArgumentParser) -> ArgumentParser:
    class CustomHelpAction(_HelpAction):
        def __call__(self, parser, namespace, values, option_string=None):
            return super().__call__(accelerate_parser, namespace, values, option_string)

    parser = ArgumentParser(
        "Accelerate launch command", add_help=False, allow_abbrev=False
    )
    parser.add_argument("-h", "--help", action=CustomHelpAction)
    parser.add_argument("--config_file", default=None, action="store", type=Path)
    parser.add_argument("script_flags", nargs=REMAINDER)
    return parser


def torch2_or_after() -> bool:
    return Version(torch.__version__) >= Version("2.0.0")


def get_user_config_from_file(user_config_file: Optional[Path]) -> Dict[str, Any]:
    user_config: Dict[str, Any] = dict()
    if user_config_file is None:
        return user_config

    if not (user_config_file.exists() and user_config_file.is_file()):
        raise ValueError(f"{user_config_file} does not exist or is not a file.")

    with user_config_file.open("r", encoding="utf-8") as file:
        if user_config_file.suffix == ".yaml":
            user_config = yaml.safe_load(file)
        elif user_config_file.suffix == ".json":
            user_config = json.load(file)
        else:
            raise NotImplementedError("Only yaml and json file can be read.")
    return user_config


def make_dist_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    default_accelerate_config: Dict[str, Any] = {
        "deepspeed_config": {},
        "fsdp_config": {},
        "main_training_function": "main",
        "megatron_lm_config": {},
        "same_network": True,
        "use_cpu": False,
    }
    mandatory_jean_zay_config: Dict[str, Any] = dict()

    processes_per_node = max(
        len(idr_torch.gpu_ids), 1
    )  # default to one process per node in case of cpu job
    num_processes = processes_per_node * idr_torch.nnodes

    distributed_config: Dict[str, Any] = {
        "machine_rank": idr_torch.rank,
        "main_process_ip": idr_torch.master_addr,
        "main_process_port": idr_torch.master_port,
        "num_machines": idr_torch.nnodes,
        "num_processes": num_processes,
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "rdzv_backend": "c10d",
        "gpu_ids": "all",
    }

    if torch2_or_after():
        default_accelerate_config.update(dynamo_backend="NO")
        mandatory_jean_zay_config.update(
            tpu_env=[],
            tpu_use_cluster=False,
            tpu_use_sudo=False,
        )

    accelerate_config: Dict[str, Any] = dict()
    accelerate_config.update(**default_accelerate_config)
    accelerate_config.update(**distributed_config)
    accelerate_config.update(**user_config)
    accelerate_config.update(**mandatory_jean_zay_config)
    return accelerate_config


def write(dist_config: Dict[str, Any]) -> Path:
    folder = Path(".accelerate_config")
    folder.mkdir(exist_ok=True)
    filename = folder / f"config_rank_{idr_torch.rank}.yaml"
    with filename.open("w") as file:
        yaml.dump(dist_config, file)
    return filename


def run() -> None:
    accelerate_parser = launch_command_parser()
    idr_parser = make_config_file_parser(accelerate_parser)
    idr_args, other_flags = idr_parser.parse_known_args()
    user_config = get_user_config_from_file(idr_args.config_file)
    dist_config = make_dist_config(user_config)
    filename = write(dist_config)
    
    print(dist_config)
    
    accelerate_flags = (
        ["--config_file", str(filename)] + other_flags + idr_args.script_flags
    )
    accelerate_args = accelerate_parser.parse_args(accelerate_flags)
    launch_command(accelerate_args)

    
if __name__ == "__main__":
    run()
