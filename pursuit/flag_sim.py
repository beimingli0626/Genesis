import argparse
import os

import genesis as gs
from flag_env import CaptureTheFlagEnv
from tqdm import tqdm

import torch
from skrl.utils.runner.torch.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-e", "--experiment_type", type=str, default="2v1_ppo")
    parser.add_argument("-l", "--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    # set device
    device = torch.device("cuda:0" if "cuda" in args.device.lower() else "cpu")

    # load config
    config_path = os.path.join("./configs", f"{args.experiment_type}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    cfg = Runner.load_cfg_from_yaml(config_path)
    cfg["agent"]["experiment"].update(
        {
            "write_interval": 0,  # disable logging, not create directory
            "checkpoint_interval": 0,  # disable checkpointing, not create directory
            "directory": "",
            "wandb": False,
        }
    )

    # init genesis
    gs.init(logging_level="error")

    # create environment
    env = CaptureTheFlagEnv(cfg=cfg["env_cfg"], show_viewer=args.vis, device=device)

    # create runner
    runner = Runner(cfg=cfg, env=env)
    runner._agent.load(
        os.path.join(
            args.log_dir, args.experiment_type, cfg["agent"]["eval"]["experiment_name"], "checkpoints", "agent_2500.pt"
        )
    )

    # run inference
    states, _ = env.reset()
    for _ in tqdm(range(1000)):
        actions = runner._agent.inference(states)
        states, _, _, _, _ = runner._env.step(actions)


if __name__ == "__main__":
    main()
