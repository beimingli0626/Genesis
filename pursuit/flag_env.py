import math
import torch
import genesis as gs

AGENT_COLORS = {
    "RED": (1.0, 0.0, 0.0),  # red
    "BLUE": (0.0, 0.0, 1.0),  # blue
}


class CaptureTheFlagEnv:
    def __init__(self, cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.cfg = cfg

        # simulation settings
        self.dt = self.cfg.get("dt", 0.01)  # run in 100hz, default
        self.step_dt = self.cfg.get("step_dt", 0.2)  # multiple low-level steps = 1 high-level step
        self.episode_length_s = self.cfg.get("episode_length_s", 5)
        self.max_episode_length = math.ceil(self.episode_length_s / self.step_dt)
        self.control_steps = int(self.step_dt / self.dt)  # number of low-level steps for 1 high-level step

        # environment settings
        self.num_envs = self.cfg.get("num_envs", 2)
        self.num_agents = self.cfg.get("agent", {}).get("num_agents", 1)  # number of agents per team
        self.num_observations = self.cfg.get("agent", {}).get("num_observations", 3)
        self.num_actions = self.cfg.get("agent", {}).get("num_actions", 3)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.cfg.get("max_visualize_FPS", 100),
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add red team (spheres)
        self.R_agents = []
        for i in range(self.num_agents):
            agent = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=AGENT_COLORS["RED"],
                    ),
                ),
            )
            self.R_agents.append(agent)

        # add blue team (spheres)
        self.B_agents = []
        for i in range(self.num_agents):
            agent = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=AGENT_COLORS["BLUE"],
                    ),
                ),
            )
            self.B_agents.append(agent)

        # visualize arena
        self.arena_size = self.cfg.get("arena", {}).get("arena_size", 2)
        # self.visualize_arena()

        # add camera
        if self.cfg.get("visualize_camera", False):
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # build scene
        self.scene.build(n_envs=self.num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_scales = self.cfg.get("reward", {}).get("reward_scales", {})
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.step_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # pursuit evasion settings
        self.tag_threshold = self.cfg.get("agent", {}).get("tag_threshold", 0.5)
        self.clip_agent_actions = self.cfg.get("agent", {}).get("clip_agent_actions", 2.0)

        # initialize buffers, note that all below buffers should be rewrite as dict for multi-agent env, with key as agent id
        self.observation = torch.zeros((self.num_envs, self.num_observations), device=self.device, dtype=gs.tc_float)
        self.rewards = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.terminated = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.truncated = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.agent_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)

        # initialize buffers need to be reset in reset()
        self.R_agents_pos = torch.zeros((self.num_envs, self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.B_agents_pos = torch.zeros((self.num_envs, self.num_agents, 3), device=self.device, dtype=gs.tc_float)

        self.episode_length = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.extras = dict()  # extra information for logging

    def step(self, actions):
        pass

    def reset_idx(self, envs_idx):
        """Reset environments

        Resample initial positions for agent and target

        Args:
            envs_idx: indices of environments to reset
        """
        if len(envs_idx) == 0:
            return

        # sample initial positions for agents and target
        if self.use_arena:
            pos_range = self.init_dist
            self.target_init_pos = torch.empty((len(envs_idx), 3), device=self.device).uniform_(-pos_range, pos_range)
            self.agent_init_pos = torch.empty((len(envs_idx), self.num_agents, 3), device=self.device).uniform_(
                -pos_range, pos_range
            )

        self.target_init_pos[..., 2] = 0.0  # Set z-coordinate to 1.0
        self.agent_init_pos[..., 2] = 0.0  # Set z-coordinate to 1.0

        # reset agents
        self.agent_pos[envs_idx] = self.agent_init_pos
        for i in range(self.num_agents):
            self.agents[i].set_pos(self.agent_pos[envs_idx, i], zero_velocity=True, envs_idx=envs_idx)
            self.agents[i].zero_all_dofs_velocity(envs_idx)

        # reset target
        self.target_pos[envs_idx] = self.target_init_pos
        self.target.set_pos(self.target_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target.zero_all_dofs_velocity(envs_idx)

    def reset(self):
        pass

    # ------------ reward functions ----------------
    def _reward_tag(self):
        """Reward for tagging the target

        Returns:
            reward: 1 if target is tagged, 0 otherwise
        """
        # TODO: implement tag reward
        return 0 < self.at_target_threshold  # distance in xy within threshold = capture

    # ------------ skrl required interface ----------------
    def state(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    # # TODO: define observations spaces, action spaces, shared observation spaces
    # # ------------ multi-agent env required properties ------------
    # @property
    # def observation_spaces(self):
    #     return self.num_observations

    # @property
    # def action_spaces(self):
    #     return self.num_actions

    # ------------ single-agent env required properties ------------
    @property
    def state_space(self):
        # TODO: change to actual state space
        # for multiagent env, this is the global state space
        return self.R_agents_pos.shape[-1]

    @property
    def observation_space(self):
        return self.num_observations

    @property
    def action_space(self):
        return self.num_actions
