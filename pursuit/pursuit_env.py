import math
import torch
import genesis as gs
import numpy as np
import gymnasium as gym

class PursuitEnv:
    def __init__(self, cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.cfg = cfg
        
        # simulation settings
        self.dt = self.cfg.get("dt", 0.01)  # run in 100hz
        self.episode_length_s = self.cfg.get("episode_length_s", 5)
        self.max_episode_length = math.ceil(self.episode_length_s / self.dt)
        
        # environment settings
        self.num_envs = self.cfg.get("num_envs", 2)
        self.num_agents = self.cfg.get("agent", {}).get("num_agents", 1)
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

        # add drone
        self.agent = self.scene.add_entity(
            morph=gs.morphs.Drone(
                file="urdf/drones/cf2x.urdf", 
                fixed=True,
                collision=True,
            ),
        )

        # add target / evader (rigid body)
        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.05,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.5, 0.5),
                ),
            ),
        )

        # visualize arena
        self.arena_size = self.cfg.get("arena", {}).get("arena_size", 2)
        self.visualize_arena()
        
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
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            
        # pursuit evasion settings
        self.debug_viz = self.cfg.get("debug_viz", False)
        self.at_target_threshold = self.cfg.get("agent", {}).get("at_target_threshold", 0.2)
        self.clip_agent_actions = self.cfg.get("agent", {}).get("clip_agent_actions", 1.5)
        self.clip_target_actions = self.cfg.get("target", {}).get("clip_target_actions", 3.0)
        
        # initialize buffers
        self.observation = torch.zeros((self.num_envs, self.num_observations), device=self.device, dtype=gs.tc_float)
        self.rewards = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.terminated = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.truncated = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        self.episode_length = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        self.agent_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_agent_actions = torch.zeros_like(self.agent_actions)  # used for smooth reward
        
        # initialize buffers need to be reset in reset()
        self.agent_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.rel_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_agent_pos = torch.zeros_like(self.agent_pos)
        self.last_target_pos = torch.zeros_like(self.target_pos)
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
        self.extras = dict()  # extra information for logging
    
    @property
    def observation_space(self):
        return self.num_observations
    
    @property
    def action_space(self):
        return self.num_actions
    
    def step(self, actions):
        """Step environment
        
        Args:
            actions: agent/pursuer action
            
        Return:
            observation: Tensor containing the observations of the environment (for the next time step)
            rewards: Tensor containing the rewards (for the current step), shape [num_envs, 1]
            terminated: Tensor indicating which environments have terminated, shape [num_envs, 1]
            truncated: Tensor indicating which environments have been truncated, shape [num_envs, 1]
            extras: Dictionary containing extra information for logging
        """       
        
        # apply agent action
        self.agent_actions = torch.clip(actions, -self.clip_agent_actions, self.clip_agent_actions)
        self.agent_actions[:, 2] = 0.0 # keep Z constant
        new_agent_pos = self.agent_pos + self.dt * self.agent_actions
        self.agent.set_pos(new_agent_pos, zero_velocity=True)
        
        # calculate and apply target action
        target_actions = self.get_repulsive_action()
        self.target_actions = torch.clip(target_actions, -self.clip_target_actions, self.clip_target_actions)
        new_target_pos = self.target_pos + self.dt * self.target_actions
        self.target.set_pos(new_target_pos, zero_velocity=True)
        
        # step genesis scene
        self.scene.step()
        
        # update buffers
        self.episode_length.add_(1)
        self.last_agent_pos[:] = self.agent_pos[:]
        self.last_target_pos[:] = self.target_pos[:]
        self.last_rel_pos[:] = self.rel_pos[:]
        self.agent_pos[:] = self.agent.get_pos()
        self.target_pos[:] = self.target.get_pos()
        self.rel_pos = self.target_pos - self.agent_pos

        # compute observations
        self.compute_observations()

        # compute reward
        self.rewards[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rewards += rew
            self.episode_sums[name] += rew
            
        #check truncate and reset
        self.terminated = (
            (torch.norm(self.rel_pos[:, :2], dim=1) < self.at_target_threshold)  # distance in xy within threshold = capture
        )
        self.truncated = (self.episode_length > self.max_episode_length)

        # place before reset_idx because it overwrites last_agent_actions for finished environments for correct reward computation at the next step
        self.last_agent_actions[:] = self.agent_actions[:]
                
        # reset
        reset_flag = self.truncated | self.terminated
        self.reset_idx(reset_flag.nonzero(as_tuple=False).flatten())

        # unsqueeze rewards, terminated, truncated to match skrl format, [num_envs, 1]
        return self.observation, self.rewards.unsqueeze(-1), self.terminated.unsqueeze(-1), self.truncated.unsqueeze(-1), self.extras
    
    def compute_observations(self):
        """Compute observations
        
        Note that observation should be deepcopy, otherwise it will point to the same memory address as rel_pos
        """
        self.observation[:] = self.rel_pos[:]
    
    def reset_idx(self, envs_idx):
        """Reset environments
        
        Resample initial positions for agent and target
        
        Args:
            envs_idx: indices of environments to reset
        """
        if len(envs_idx) == 0:
            return

        # sample initial positions for agent and target
        range = 1.5
        self.agent_init_pos = torch.ones((len(envs_idx), 3), device=self.device)
        self.agent_init_pos[:, :2] = torch.empty((len(envs_idx), 2), device=self.device).uniform_(-range, range)
        self.target_init_pos = torch.ones((len(envs_idx), 3), device=self.device)
        self.target_init_pos[:, :2] = torch.empty((len(envs_idx), 2), device=self.device).uniform_(-range, range)
        
        # reset agent
        self.agent_pos[envs_idx] = self.agent_init_pos
        self.agent.set_pos(self.agent_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.agent.zero_all_dofs_velocity(envs_idx)

        # reset target
        self.target_pos[envs_idx] = self.target_init_pos
        self.target.set_pos(self.target_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target.zero_all_dofs_velocity(envs_idx)
        
        self.rel_pos[envs_idx] = self.target_pos[envs_idx] - self.agent_pos[envs_idx]
        
        self.last_agent_actions[envs_idx] = 0  # initial last action should be zero for a new episode
        self.episode_length[envs_idx] = 0
        for key in self.episode_sums.keys():
            self.episode_sums[key][envs_idx] = 0.0
        
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.observation, None
        
    def get_repulsive_action(self):
        """Compute repulsive action for the evader
        
        The closer pursuer is to the evader, the stronger the force, refer to https://arxiv.org/pdf/2010.08193
        
        Returns:
            force: repulsive action
        """        
        # pursuer force
        force_pursuer = self.target_pos - self.agent_pos
        force_pursuer[:, 2] = 0
        norm = torch.norm(force_pursuer, dim=1, keepdim=True) ** 2 + 1e-5
        force_pursuer = force_pursuer / norm
        
        # arena force, receive very large force when out of arena
        force_arena = torch.zeros_like(force_pursuer)
        target_origin_dist = torch.norm(self.target_pos[..., :2], dim=-1, keepdim=True)
        force_arena_vec = -self.target_pos[..., :2] / (target_origin_dist + 1e-5)   # unit vector pointing to center
        out_of_arena = torch.norm(self.target_pos[..., :2], dim=-1, keepdim=True) > self.arena_size
        force_arena[..., :2] = out_of_arena.float() * force_arena_vec * (1 / 1e-5) + \
            (~out_of_arena).float() * force_arena_vec * (1 / ((self.arena_size - target_origin_dist) + 1e-5))
        
        force = force_pursuer + force_arena
        
        # visualize forces for debugging
        if self.debug_viz:
            self.visualize_forces(force_pursuer, force_arena, force)
        
        return force

    # ------------ reward functions ----------------
    def _reward_target(self):
        """Reward for moving towards the target
        
        Reward is the difference in the square of distance to target between current and previous step
        NOTE: this is not a good reward function, it is just for testing; it will drive the agent to 
        stay within certain distance from the target (why?)
        """
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        """Reward for smooth action
        
        Reward is the square of difference in agent action between current and previous step
        """
        smooth_rew = torch.sum(torch.square(self.agent_actions - self.last_agent_actions), dim=1)
        return smooth_rew
    
    def _reward_capture(self):
        """Reward for capturing the target
        
        Returns:
            reward: 1 if target is captured, 0 otherwise
        """
        return (torch.norm(self.rel_pos[:, :2], dim=1) < self.at_target_threshold)  # distance in xy within threshold = capture


    # ------------ debug visualization ----------------
    def visualize_forces(self, force_pursuer, force_arena, total_force, env_idx=0):
        """Visualize the forces acting on the target using debug arrows
        
        Args:
            force_pursuer: Repulsive force from pursuer
            force_arena: Force from arena boundary
            total_force: Combined total force
            env_idx: Index of environment to visualize (default 0)
        """
        # Clear previous debug objects
        self.scene.clear_debug_objects()
        
        # Get position for selected environment
        pos = self.target_pos[env_idx].cpu().numpy()
        
        # Draw arrows for each force component
        self.scene.draw_debug_arrow(
            pos=pos,
            vec=force_arena[env_idx].cpu().numpy(),
            radius=0.01,
            color=(1.0, 0.0, 0.0, 0.5)      # Red for arena force
        )
        self.scene.draw_debug_arrow(
            pos=pos,
            vec=force_pursuer[env_idx].cpu().numpy(),
            radius=0.01,
            color=(0.0, 1.0, 0.0, 0.5)      # Green for pursuer force
        )
        self.scene.draw_debug_arrow(
            pos=pos,
            vec=total_force[env_idx].cpu().numpy(),
            radius=0.01,
            color=(0.0, 0.0, 1.0, 0.5)      # Blue for total force
        )

    def visualize_arena(self):
        """Draw a thin cylinder to represent the circle boundary
        """
        self.arena_circle = self.scene.add_entity(
            morph=gs.morphs.Cylinder(
                radius=self.arena_size,
                height=0.01,
                fixed=True,
                visualization=True,
                collision=False
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 0.0, 0.0, 0.5), # Black color with 0.5 alpha
            ),
        )
        
    # ------------ skrl required interface ----------------
    def state(self):
        pass
    
    def render(self):
        pass
    
    def close(self):
        pass