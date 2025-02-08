import argparse
import os
import shutil
from datetime import datetime
import yaml

import genesis as gs
from flag_env import CaptureTheFlagEnv

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from wandb_config import WANDB_API_KEY, WANDB_ENTITY

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

set_seed()  # random seed
# set_seed(42) # for fixed seed


def get_configs(log_dir, experiment_type, experiment_name):
    # Load config from YAML file
    config_path = os.path.join("./configs", f"{experiment_type}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg_dict = config["train_cfg"]
    env_cfg_dict = config["env_cfg"]

    # Add dynamic configuration that shouldn't be in YAML
    train_cfg_dict["experiment"].update(
        {
            "directory": os.path.join(log_dir, experiment_type),
            "experiment_name": experiment_name,
            "wandb_kwargs": {
                "project": "capture-the-flag",
                "entity": WANDB_ENTITY,
                "group": experiment_type,
            },
        }
    )

    if train_cfg_dict["learning_rate_scheduler"] == "KLAdaptiveRL":
        train_cfg_dict["learning_rate_scheduler"] = KLAdaptiveRL
    else:
        train_cfg_dict["learning_rate_scheduler"] = None

    return train_cfg_dict, env_cfg_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-e", "--experiment_type", type=str, default="1v1_ppo_open")
    parser.add_argument("-l", "--log_dir", type=str, default="logs")
    args = parser.parse_args()

    # create experiment directory
    experiment_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    log_dir = os.path.join(args.log_dir, args.experiment_type, experiment_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # set device
    if args.device == "cuda:0" or args.device == "cuda" or args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # get configs
    train_cfg_dict, env_cfg = get_configs(args.log_dir, args.experiment_type, experiment_name)
    train_cfg = PPO_DEFAULT_CONFIG.copy()
    train_cfg.update(train_cfg_dict)
    train_cfg["env_cfg"] = env_cfg  # this will be logged to wandb

    # init genesis
    gs.init(logging_level="")
    # gs.init(logging_level="error")

    # create environment
    env = CaptureTheFlagEnv(cfg=env_cfg, show_viewer=args.vis, device=device)

    _, _ = env.reset()
    for i in range(1000):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.device))
