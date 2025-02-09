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

        # environment settings
        self.num_envs = self.cfg.get("num_envs", 1024)
        self.num_agents = self.cfg.get("agent", {}).get("num_agents", 1)  # number of agents per team
        self.num_obs = self.cfg.get("agent", {}).get("num_observations", 3) * self.num_agents  # 3 for each agent
        self.num_actions = (
            self.cfg.get("agent", {}).get("num_actions", 3) * self.num_agents
        )  # 3 for each agent, pursuer team
        self.num_privileged_obs = None

        # simulation settings
        self.dt = self.cfg.get("dt", 0.01)  # run in 100hz, default
        self.step_dt = self.cfg.get("step_dt", 0.2)  # multiple low-level steps = 1 high-level step
        self.episode_length_s = self.cfg.get("episode_length_s", 5)
        self.max_episode_length = math.ceil(self.episode_length_s / self.step_dt)
        self.control_steps = int(self.step_dt / self.dt)  # number of low-level steps for 1 high-level step

        # pursuit evasion settings
        self.tag_threshold = self.cfg.get("agent", {}).get("tag_threshold", 0.5)
        self.clip_agent_actions = self.cfg.get("agent", {}).get("clip_agent_actions", 2.0)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.cfg.get("max_visualize_FPS", 100),
                camera_pos=(0.0, 0.0, 5.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                n_rendered_envs=1,
            ),
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
        self.size_agent = self.cfg.get("agent", {}).get("size_agent", 0.05)
        self.R_agents = []
        for i in range(self.num_agents):
            agent = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=self.size_agent,
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
                    scale=self.size_agent,
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

        # generate arena
        self.arena_size = self.cfg.get("arena", {}).get("arena_size", 2)
        self.generate_arena()

        # build scene
        self.scene.build(n_envs=self.num_envs)

        # add center line visualization using debug line
        line_height = 0.01  # Slightly above ground to be visible
        self.center_line = self.scene.draw_debug_line(
            start=(0, self.arena_size / 2, line_height),  # Start at north wall
            end=(0, -self.arena_size / 2, line_height),  # End at south wall
            radius=0.01,  # Line thickness
            color=(1.0, 1.0, 1.0, 0.5),  # White color with 50% transparency
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_scales = self.cfg.get("reward", {}).get("reward_scales", {})
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.step_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers, note that all below buffers should be rewrite as dict for multi-agent env, with key as agent id
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.terminated = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.truncated = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        self.R_actions = torch.zeros(
            (self.num_envs, self.num_agents, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.B_actions = torch.zeros(
            (self.num_envs, self.num_agents, self.num_actions), device=self.device, dtype=gs.tc_float
        )

        # initialize buffers need to be reset in reset()
        self.R_pos = torch.zeros((self.num_envs, self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.B_pos = torch.zeros((self.num_envs, self.num_agents, 3), device=self.device, dtype=gs.tc_float)
        self.min_dist = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.episode_length = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.extras = {"observations": {}}  # extra information for logging

    def generate_arena(self):
        """Generate square arena with walls"""
        wall_height = 0.01  # Height of the walls
        wall_thickness = 0.01  # Thickness of the walls
        half_size = self.arena_size / 2

        # wall configurations: (name, position, size)
        wall_configs = [
            ("north", (0, half_size, wall_height / 2), (self.arena_size + wall_thickness, wall_thickness, wall_height)),
            (
                "south",
                (0, -half_size, wall_height / 2),
                (self.arena_size + wall_thickness, wall_thickness, wall_height),
            ),
            ("east", (half_size, 0, wall_height / 2), (wall_thickness, self.arena_size + wall_thickness, wall_height)),
            ("west", (-half_size, 0, wall_height / 2), (wall_thickness, self.arena_size + wall_thickness, wall_height)),
        ]

        # create walls
        self.walls = {}
        for name, pos, size in wall_configs:
            self.walls[name] = self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=pos,
                    size=size,
                    fixed=True,
                    collision=True,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 1.0, 1.0)),
                ),
            )

    def step(self, actions):
        clipped_actions = torch.clip(actions, -self.clip_agent_actions, self.clip_agent_actions)
        self.R_actions = clipped_actions.view(self.num_envs, self.num_agents, 3)
        self.R_actions[..., 2] = 0.0  # keep Z constant for all agents

        self.B_actions = torch.randn_like(self.R_actions)  # random move
        self.B_actions[..., 2] = 0.0  # keep Z constant for all agents

        for _ in range(self.control_steps):
            # Update positions for all agents
            for i in range(self.num_agents):
                self.R_pos[:, i] = self.R_agents[i].get_pos()
                new_R_pos = self.get_new_pos(self.R_pos[:, i], self.R_actions[:, i])
                self.R_agents[i].set_pos(new_R_pos, zero_velocity=True)
                self.R_agents[i].zero_all_dofs_velocity()

            # calculate and apply target action
            for i in range(self.num_agents):
                self.B_pos[:, i] = self.B_agents[i].get_pos()
                new_B_pos = self.get_new_pos(self.B_pos[:, i], self.B_actions[:, i])
                self.B_agents[i].set_pos(new_B_pos, zero_velocity=True)
                self.B_agents[i].zero_all_dofs_velocity()

            # step genesis scene
            self.scene.step()

        # update buffers
        self.episode_length.add_(1)

        # query agent positions
        for i in range(self.num_agents):
            self.R_pos[:, i] = self.R_agents[i].get_pos()

        for i in range(self.num_agents):
            self.B_pos[:, i] = self.B_agents[i].get_pos()

        if self.num_agents == 1:
            rel_pos = self.R_pos - self.B_pos  # [num_envs_idx, 1, 3]
            self.min_dist = torch.norm(rel_pos, dim=-1).squeeze(-1)  # [num_envs_idx]

        # check truncate and reset
        # self.terminated = self.min_dist < self.at_target_threshold
        self.truncated = self.episode_length > self.max_episode_length
        time_out_idx = (self.episode_length > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # reset
        self.reset_buf = self.truncated | self.terminated
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.get_observations()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        R_pos_flat = self.R_pos.reshape(self.num_envs, -1)  # [num_envs_idx, num_R_agents * 3]
        B_pos_flat = self.B_pos.reshape(self.num_envs, -1)  # [num_envs_idx, num_B_agents * 3]
        self.obs_buf = torch.cat([R_pos_flat, B_pos_flat], dim=1)  # [num_envs_idx, (num_R_agents + num_B_agents) * 3]
        return self.obs_buf, self.extras

    def reset_idx(self, envs_idx):
        """Reset environments

        Initialize red team on negative x-side and blue team on positive x-side of the arena

        Args:
            envs_idx: indices of environments to reset
        """
        if len(envs_idx) == 0:
            return

        # Calculate position ranges for each team
        half_arena = self.arena_size / 2

        # Red team: negative x side (-arena_size/2 to 0)
        R_pos_x = torch.empty((len(envs_idx), self.num_agents), device=self.device).uniform_(
            -half_arena + self.size_agent, -self.size_agent
        )
        R_pos_y = torch.empty((len(envs_idx), self.num_agents), device=self.device).uniform_(
            -half_arena + self.size_agent, half_arena - self.size_agent
        )

        # Blue team: positive x side (0 to arena_size/2)
        B_pos_x = torch.empty((len(envs_idx), self.num_agents), device=self.device).uniform_(
            self.size_agent, half_arena - self.size_agent
        )
        B_pos_y = torch.empty((len(envs_idx), self.num_agents), device=self.device).uniform_(
            -half_arena + self.size_agent, half_arena - self.size_agent
        )

        # Combine positions into 3D coordinates (z=0 for ground plane)
        self.R_pos[envs_idx] = torch.stack([R_pos_x, R_pos_y, torch.zeros_like(R_pos_x)], dim=-1)
        self.B_pos[envs_idx] = torch.stack([B_pos_x, B_pos_y, torch.zeros_like(B_pos_x)], dim=-1)

        # Set positions for red team agents
        for i in range(self.num_agents):
            self.R_agents[i].set_pos(self.R_pos[envs_idx, i], zero_velocity=True, envs_idx=envs_idx)
            self.R_agents[i].zero_all_dofs_velocity(envs_idx)

        # Set positions for blue team agents
        for i in range(self.num_agents):
            self.B_agents[i].set_pos(self.B_pos[envs_idx, i], zero_velocity=True, envs_idx=envs_idx)
            self.B_agents[i].zero_all_dofs_velocity(envs_idx)

        # if self.num_agents == 1:
        #     rel_pos = self.R_pos[envs_idx] - self.B_pos[envs_idx]  # [num_envs_idx, 1, 3]
        #     self.min_dist[envs_idx] = torch.norm(rel_pos, dim=-1).squeeze(-1)  # [num_envs_idx]

        self.episode_length[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        for key in self.episode_sums.keys():
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, self.extras

    def get_new_pos(self, current_pos, action):
        """Calculate new position while enforcing arena boundaries

        Args:
            current_pos: Current position tensor [num_envs, 3]
            action: Action tensor [num_envs, 3]

        Returns:
            new_pos: New position tensor [num_envs, 3] clipped to arena bounds
        """
        new_pos = current_pos + self.dt * action
        half_size = self.arena_size / 2
        new_pos[..., 0] = torch.clip(new_pos[..., 0], -half_size + self.size_agent, half_size - self.size_agent)
        new_pos[..., 1] = torch.clip(new_pos[..., 1], -half_size + self.size_agent, half_size - self.size_agent)
        return new_pos

    # ------------ reward functions ----------------
    def _reward_tag(self):
        """Reward for tagging the target

        Returns:
            reward: 1 if target is tagged, -1 otherwise
        """
        return torch.where(
            self.min_dist <= self.tag_threshold, torch.ones_like(self.min_dist), -torch.ones_like(self.min_dist)
        )

    # ------------ rsl_rl required interface ----------------
    def get_privileged_observations(self):
        return None
