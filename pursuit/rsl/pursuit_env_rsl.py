import math
import torch
import genesis as gs
import numpy as np


class PursuitEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg=None, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]

        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
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

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

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
        self.arena_size = env_cfg["arena_size"]
        self.visualize_arena()
        
        # build scene
        self.scene.build(n_envs=num_envs)
        
        # prepare reward functions and multiply reward scales by dt
        self.reward_scales = reward_cfg["reward_scales"]
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            
        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        self.agent_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_agent_actions = torch.zeros_like(self.agent_actions)
        
        self.agent_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.agent_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.rel_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_agent_pos = torch.zeros_like(self.agent_pos)
        self.last_target_pos = torch.zeros_like(self.target_pos)
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
        
        self.extras = dict()  # extra information for logging
        self.finish_condition = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        
        # pursuit evasion settings
        self.debug_viz = env_cfg.get("debug_viz", False)
    
    def step(self, actions):
        """Step environment
        
        Args:
            actions: agent/pursuer action
            
        Return:
            - obs_buf: Tensor containing the observations of the environment (for the next time step).
            - None: Placeholder for privileged observations (not used here).
            - rew_buf: Tensor containing the rewards (for the current step).
            - reset_buf: Tensor indicating which environments need to be reset.
            - extras: Dictionary containing extra information for logging.
        """       
        
        # apply target action
        target_actions = self.get_repulsive_action()
        self.target_actions = torch.clip(target_actions, -self.env_cfg["clip_target_actions"], self.env_cfg["clip_target_actions"])
        new_target_pos = self.target_pos + self.dt * self.target_actions
        self.target.set_pos(new_target_pos, zero_velocity=True)
        
        # apply agent action, replace this with actual prediction of policy network
        self.agent_actions = torch.clip(actions, -self.env_cfg["clip_agent_actions"], self.env_cfg["clip_agent_actions"])
        self.agent_actions[:, 2] = 0.0 # keep Z constant
        new_agent_pos = self.agent_pos + self.dt * self.agent_actions
        self.agent.set_pos(new_agent_pos, zero_velocity=True)
        
        # step genesis scene
        self.scene.step()
        
        # update buffers
        self.episode_length_buf += 1
        self.last_agent_pos[:] = self.agent_pos[:]
        self.last_target_pos[:] = self.target_pos[:]
        self.last_rel_pos = self.rel_pos
        self.agent_pos[:] = self.agent.get_pos()
        self.target_pos[:] = self.target.get_pos()
        self.rel_pos = self.target_pos - self.agent_pos

        # compute observations
        self.compute_observations()
        
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            
        #check termination and reset
        self.finish_condition = (
            (torch.norm(self.rel_pos[:, :2], dim=1) < self.env_cfg["at_target_threshold"])  # distance in xy within threshold = capture
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.finish_condition
        
        # time out
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # reset
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        self.last_agent_actions[:] = self.agent_actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras
    
    def compute_observations(self):
        self.obs_buf = self.rel_pos
    
    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None
    
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # sample initial positions for agent and target
        range = 2.0
        self.agent_init_pos = torch.ones((len(envs_idx), 3), device=self.device)
        self.agent_init_pos[:, :2] = torch.empty((len(envs_idx), 2), device=self.device).uniform_(-range, range)
        self.target_init_pos = 0.5 * torch.ones((len(envs_idx), 3), device=self.device)
        self.target_init_pos[:, :2] = torch.empty((len(envs_idx), 2), device=self.device).uniform_(-range, range)
        
        # reset agent
        self.agent_pos[envs_idx] = self.agent_init_pos
        self.agent.set_pos(self.agent_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.agent.zero_all_dofs_velocity(envs_idx)
        # self.agent_quat[envs_idx] = self.agent_init_quat.reshape(1, -1) # [num_env, 4]

        # reset target
        self.target_pos[envs_idx] = self.target_init_pos
        self.target.set_pos(self.target_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.target.zero_all_dofs_velocity(envs_idx)
        
        self.rel_pos[envs_idx] = self.target_pos[envs_idx] - self.agent_pos[envs_idx]
        self.last_rel_pos[envs_idx] = self.rel_pos[envs_idx]
    
        # reset buffers
        self.last_agent_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        for key in self.episode_sums.keys():
            self.episode_sums[key][envs_idx] = 0.0
        
    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None
        
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
        """
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        """Reward for smooth action
        
        Reward is the square of difference in agent action between current and previous step
        """
        smooth_rew = torch.sum(torch.square(self.agent_actions - self.last_agent_actions), dim=1)
        return smooth_rew
    
    # def _reward_capture(self):
    #     """Reward for capturing the target
        
    #     Returns:
    #         reward: 1 if target is captured, 0 otherwise
    #     """
    #     return (torch.norm(self.rel_pos[:, :2], dim=1) < self.env_cfg["at_target_threshold"])  # distance in xy within threshold = capture


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