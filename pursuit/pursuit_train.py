import argparse
import os
import pickle
import shutil
from datetime import datetime

import genesis as gs
from pursuit_env import PursuitEnv

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


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def get_train_cfg(log_dir, experiment_name):
    train_cfg_dict = {
        "rollouts": 10,  # number of rollouts before updating
        "learning_epochs": 5,  # number of learning epochs during each update
        "mini_batches": 4,  # number of mini batches during each learning epoch
        "discount_factor": 0.99,  # discount factor (gamma)
        "lambda": 0.95,  # TD(lambda) coefficient (lam) for computing returns and advantages
        "learning_rate": 3e-4,  # learning rate
        "learning_rate_scheduler": KLAdaptiveRL,  # learning rate scheduler class
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.01, "min_lr": 1e-5},  # learning rate scheduler's kwargs
        "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
        "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
        "value_preprocessor": None,  # value preprocessor class (see skrl.resources.preprocessors)
        "value_preprocessor_kwargs": {},  # value preprocessor's kwargs (e.g. {"size": 1})
        "random_timesteps": 0,  # random exploration steps
        "learning_starts": 0,  # learning starts after this many steps
        "grad_norm_clip": 1.0,  # clipping coefficient for the norm of the gradients
        "ratio_clip": 0.2,  # clipping coefficient for computing the clipped surrogate objective
        "value_clip": 0.2,  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
        "clip_predicted_values": True,  # clip predicted values during value loss computation
        "entropy_loss_scale": 0.01,  # entropy loss scaling factor
        "value_loss_scale": 1.0,  # value loss scaling factor
        "kl_threshold": 0.0,  # KL divergence threshold for early stopping, prevent overfitting
        "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
        "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)
        "mixed_precision": True,  # enable automatic mixed precision for higher performance
        "experiment": {
            "directory": log_dir,  # experiment's parent directory
            "experiment_name": experiment_name,  # experiment name
            "write_interval": 40,  # TensorBoard writing interval (timesteps), "auto" save 100 records
            "checkpoint_interval": "auto",  # interval for checkpoints (timesteps), "auto" save 10 checkpoints
            "store_separately": False,  # whether to store checkpoints separately
            "wandb": False,  # whether to use Weights & Biases
            "wandb_kwargs": {  # wandb kwargs
                "project": "pursuit-evasion",
                "entity": WANDB_ENTITY,
                "group": "1v1_pursuit",
            },
        },
    }
    return train_cfg_dict


def get_env_cfg():
    env_cfg_dict = {
        "num_envs": 1024,
        "episode_length_s": 5.0,
        "dt": 0.01,
        # agent
        "agent": {
            "num_agents": 1,
            "num_observations": 3,  # number of observations per agent
            "num_actions": 3,  # number of actions per agent
            "at_target_threshold": 0.5,
            "clip_agent_actions": 3.0,
            "observation_mode": ["rel_pos"],
            "collision_threshold": 0.2,
        },
        "target": {
            "clip_target_actions": 2.5,
        },
        "arena": {
            "use_arena": False,
            "arena_size": 2.0,
        },
        "reward": {
            "reward_scales": {
                "distance": -1,
                # "target": 10.0,
                # "smooth": 1e-4,  # two training phase, only consider smooth reward in the second phase
                "capture": 1.0,  # capture 100 is too high, make the agent overfit to move to one single direction
                # capture 10 is too high for two agents with arena
                # "collision": -10,
            },
        },
        # visualization
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        # debug visualization
        "debug_viz": True,
    }
    return env_cfg_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-l", "--log_dir", type=str, default="logs/1v1_pursuit")
    args = parser.parse_args()

    # create experiment directory
    experiment_name = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    log_dir = os.path.join(args.log_dir, experiment_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # set device
    if args.device == "cuda:0" or args.device == "cuda" or args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # get configs
    env_cfg = get_env_cfg()
    train_cfg = PPO_DEFAULT_CONFIG.copy()
    train_cfg.update(get_train_cfg(args.log_dir, experiment_name))
    train_cfg["env_cfg"] = env_cfg  # this will be logged to wandb
    # pickle.dump([env_cfg, train_cfg], open(f"{args.log_dir}/{experiment_name}/cfgs.pkl", "wb"))

    # init genesis
    # gs.init(logging_level="")
    gs.init(logging_level="error")

    # create environment
    env = PursuitEnv(cfg=env_cfg, show_viewer=args.vis, device=device)

    # create policy and value models
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device=device)
    models["value"] = Value(env.observation_space, env.action_space, device=device)

    # create memory
    memory = RandomMemory(memory_size=train_cfg["rollouts"], num_envs=env.num_envs, device=device)

    # create agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=train_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # create trainer, note that genesis is agnostic to "headless"
    cfg_trainer = {"timesteps": 16000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # train
    trainer.train()


if __name__ == "__main__":
    main()
