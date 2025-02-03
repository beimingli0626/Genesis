import argparse
import os
import pickle
import shutil
from datetime import datetime

import genesis as gs
from pursuit_env_rsl import PursuitEnv
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


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
            # "capture": 1.0,
        },
    }
    command_cfg = {}

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="pursuit")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=300)
    args = parser.parse_args()

    gs.init(logging_level="error")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{args.exp_name}/{timestamp}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env = PursuitEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
