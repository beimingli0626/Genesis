import argparse
import numpy as np
import genesis as gs
from pursuit_env import PursuitEnv


def get_cfgs():
    env_cfg = {
        "num_actions": 3,
        "episode_length_s": 5.0,
        # agent pose
        "at_target_threshold": 0.2,
        "clip_agent_actions": 1.5,
        # target pose
        "clip_target_actions": 3.0,
        # arena
        "arena_size": 3.0,
        # visualization
        "visualize_camera": True,
        "max_visualize_FPS": 60,
        # debug visualization
        "debug_viz": True,
    }
    obs_cfg = {
        "num_obs": 3,    
    }
    reward_cfg = {
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
        },
    }
    command_cfg = {
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="pursuit")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=300)
    args = parser.parse_args()
    
    gs.init()

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env = PursuitEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )
    
    import torch
    _, _ = env.reset()
    for i in range(1000):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.device))


if __name__ == "__main__":
    main()
    