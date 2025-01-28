import argparse
import os
import pickle
import shutil
from datetime import datetime

import genesis as gs
from pursuit_env import PursuitEnv

import torch
import torch.nn as nn

# import skrl components
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# set seed
set_seed()  # set_seed(42) for fixed seed
        
# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

        
def get_train_cfg(log_dir):
    train_cfg_dict = {
        "rollouts": 100, 
        "learning_epochs": 5,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 3e-4,
        "learning_rate_scheduler": KLAdaptiveRL,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.0,
        "value_loss_scale": 1.0,
        "kl_threshold": 0,
        "time_limit_bootstrap": False,
        "experiment": {
            "write_interval": 40,
            "checkpoint_interval": 400,
            "directory": log_dir
        }
    }

    return train_cfg_dict

def get_cfgs():
    env_cfg = {
        "num_actions": 3,
        "episode_length_s": 5.0,
        # agent
        # "num_agents": 2,
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
        "num_observations": 3,    
    }
    reward_cfg = {
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
            # "capture": 1.0,
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
    args = parser.parse_args()
    
    gs.init()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{args.exp_name}/{timestamp}"
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
    
    models = {}
    models["policy"] = Policy(env.num_observations, env.num_actions, torch.device("cuda:0"))
    models["value"] = Value(env.num_observations, env.num_actions, torch.device("cuda:0"))
    
    train_cfg = PPO_DEFAULT_CONFIG.copy()
    
    memory = RandomMemory(memory_size=train_cfg["rollouts"], num_envs=env.num_envs, device=torch.device("cuda:0"))
    
    agent = PPO(models=models,
            memory=memory,
            cfg=train_cfg,
            observation_space=env.num_observations,
            action_space=env.num_actions,
            device=torch.device("cuda:0"))
    
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    cfg_trainer = {"timesteps": 8000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    trainer.train()
    

if __name__ == "__main__":
    main()
    