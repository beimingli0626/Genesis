import argparse
import os
import shutil
from datetime import datetime
import yaml
import torch

import genesis as gs
from flag_env import CaptureTheFlagEnv

from rsl_rl.runners import OnPolicyRunner
from wandb_config import WANDB_API_KEY, WANDB_ENTITY

os.environ["WANDB_USERNAME"] = WANDB_ENTITY
os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def get_configs(experiment_type):
    # Load config from YAML file
    config_path = os.path.join("./configs", f"{experiment_type}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-e", "--experiment_type", type=str, default="1v1_ppo")
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
    config = get_configs(args.experiment_type)

    # init genesis
    # gs.init()
    gs.init(logging_level="error")

    # create environment
    env = CaptureTheFlagEnv(cfg=config["env_cfg"], show_viewer=args.vis, device=device)
    _, _ = env.reset()

    runner = OnPolicyRunner(env, train_cfg=config["train_cfg"], log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=config["train_cfg"]["num_learning_iterations"], init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
