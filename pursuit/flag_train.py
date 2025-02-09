import argparse
import os

import genesis as gs
from flag_env import CaptureTheFlagEnv

import torch
from skrl.utils.runner.torch.runner import Runner
from wandb_config import WANDB_API_KEY, WANDB_ENTITY

os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-e", "--experiment_type", type=str, default="1v1_ppo")
    parser.add_argument("-l", "--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    # create experiment directory
    log_dir = os.path.join(args.log_dir, args.experiment_type)
    os.makedirs(log_dir, exist_ok=True)

    # set device
    device = torch.device("cuda:0" if "cuda" in args.device.lower() else "cpu")

    # load config
    config_path = os.path.join("./configs", f"{args.experiment_type}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    cfg = Runner.load_cfg_from_yaml(config_path)
    cfg["agent"]["experiment"].update(
        {
            "directory": log_dir,
            "wandb": True,
            "wandb_kwargs": {
                "project": "capture-the-flag",
                "entity": WANDB_ENTITY,
                "group": args.experiment_type,
            },
        }
    )

    # init genesis
    gs.init(logging_level="error")

    # create environment
    env = CaptureTheFlagEnv(cfg=cfg["env_cfg"], show_viewer=args.vis, device=device)

    # create runner
    runner = Runner(cfg=cfg, env=env)

    # run training
    runner.run("train")


if __name__ == "__main__":
    main()
