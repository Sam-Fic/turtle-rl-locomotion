from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}


class TurtleMujocoEnv(MujocoEnv):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, ctrl_type="position", n_envs=1, **kwargs):
        self.n_envs = n_envs
        self._curriculum_base = 0.3
        self._step_count = 0
        self.is_training = True  # 训练模式默认开启，测试时关闭以防止课程进度被覆盖
        self._use_curriculum_progress = False  # 是否启用课程进度自动更新，关闭后进度将固定为最大值 1.0
        # 启用课程进度时从 0 开始逐步增长；关闭时固定为 1.0（满强度）
        self._curriculum_progress = 0.0 if self._use_curriculum_progress else 1.0

        self._dr_enabled = {
            "friction": False,
            "mass": False,
            "damping": False,
            "armature": False,
            "frictionloss": False,
            "motor_kp": False,
            "gravity": False,
            "sensor_noise": False,
            "action_delay": False,
            "joint_bias": False,
        }
        self._dr_scale_ranges = {
            "friction": (0.5, 1.5),
            "mass": (0.7, 1.3),
            "damping": (0.5, 1.5),
            "armature": (0.7, 1.3),
            "frictionloss": (0.0, 1.3),
            "motor_kp": (0.9, 1.1),
            "gravity_z": (-0.5, 0.5),
            "joint_bias": (0.0, 0.15),
        }
        self._dr_noise_levels = {
            "linear_velocity": 0.02,
            "angular_velocity": 0.002,
            "dofs_position": 0.005,
            "dofs_velocity": 0.05,
            "projected_gravity": 0.005,
        }
        self._dr_action_delay_range = (0, 3)
        self._action_buffer = np.zeros(
            (self._dr_action_delay_range[1] + 1, 10))
        self._action_delay = 0
        self._joint_bias = np.zeros(10)

        model_path = Path(f"./turtle/scene_{ctrl_type}.xml")
        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            # dt(=0.001) * 20 = 0.02 seconds -> 50Hz action rate
            frame_skip=20,
            observation_space=None,  # Manually set afterwards
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Update metadata to include the render FPS
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 60,
        }
        self._last_render_time = -1.0
        self._max_episode_time_sec = 15.0
        self._step = 0

        # Weights for the reward and cost functions
        self.reward_weights = {
            "linear_vel_tracking": 1.0, 
            "angular_vel_tracking": 1.0,
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 0.2,  # Was 1.0
            "xy_angular_vel": 0.05,  # Was 0.05
            "action_rate": 0.05,
            "joint_limit": 1.5,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7,
            "orientation": 0.3,
            "collision": 0.05,
            "default_joint_position": 0.5
        }

        self._gravity_vector = np.array(self.model.opt.gravity)
        # 从 key_qpos 获取关节角度，跳过前7个基座自由度（位置和姿态）
        self._default_joint_position = np.array(self.model.key_qpos[0, 7:])

        # Store base model parameters for domain randomization (avoids compounding drift across episodes)
        self._base_geom_friction = self.model.geom_friction.copy()
        self._base_body_mass = self.model.body_mass.copy()
        self._base_dof_damping = self.model.dof_damping.copy()
        self._base_dof_armature = self.model.dof_armature.copy()
        self._base_dof_frictionloss = self.model.dof_frictionloss.copy()
        self._base_actuator_gainprm = self.model.actuator_gainprm.copy()
        self._base_actuator_biasprm = self.model.actuator_biasprm.copy()
        self._base_gravity = self.model.opt.gravity.copy()

        # vx (m/s), vy (m/s), wz (rad/s)
        # Turtle: -Y direction is forward, +Y is backward
        self._desired_velocity_min = np.array([0.0, -0.5, -0.0])
        self._desired_velocity_max = np.array([0.0, -0.5, 0.0])
        self._desired_velocity = self._sample_desired_vel()  # [0.0, -0.5, 0.0]
        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        # 论文中 sigma_v 和 sigma_omega 均为 0.25
        self._tracking_velocity_sigma = 0.25
        
        # 论文能量奖励超参数
        self.alpha_en = 0.0        # 能量奖励权重：初期设为0，待智能体学会稳定步态后再开启
        self.sigma_en_x = 1000.0  # 线速度能量缩放常数
        self.sigma_en_z = 500.0   # 角速度能量缩放常数
        self.alpha_ang = 0.5      # 角速度跟踪奖励权重

        # Metrics used to determine if the episode should be terminated
        self._healthy_z_range = (0.08, 0.40)
        self._healthy_pitch_range = (-np.deg2rad(10), np.deg2rad(10))
        self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

        # Feet flippers (for landing reward): link3(4), link6(7), link8(9), link10(11)
        self._cfrc_ext_feet_indices = [4, 7, 9, 11]
        # Thigh & lower leg bodies (for collision penalty): link1(2), link2(3), link4(5), link5(6), link7(8), link9(10)
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 10]

        # Non-penalized degrees of freedom range of the control joints
        dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )
        # First value is the root joint, so we ignore it
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1

        # Action: 10 torque values (Turtle has 10 joints)
        self._last_action = np.zeros(10)
        self._last_qpos = np.zeros(
            self.model.nq - 7  # joint positions, excluding root (7 DOFs: pos+quat)
        )

        # Normalized action space for residual control
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_scale = 0.7  # Max radians to change per step from default position

        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        # Feet site names to index mapping
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-site
        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtobj
        feet_site = [
            "end_leg1",
            "end_leg2",
            "end_leg3",
            "end_leg4",
        ]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }

        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )

    def step(self, action):
        self._step += 1
        if self.is_training and self._use_curriculum_progress:  # 仅在训练模式且启用课程进度时更新
            self._step_count += 1
            self._curriculum_progress = np.clip(
                self._step_count / (10_000_000 / self.n_envs), 0.0, 1.0
            )

        self._action_buffer = np.roll(self._action_buffer, shift=1, axis=0)
        self._action_buffer[0] = action
        delayed_action = self._action_buffer[self._action_delay]

        # Residual control: map normalized action [-1, 1] to delta from default position
        real_action = self._default_joint_position + delayed_action * self.action_scale
        biased_action = np.clip(
            real_action + self._joint_bias,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        self.do_simulation(biased_action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        # TODO: Consider terminating if knees touch the ground
        terminated = not self.is_healthy
        truncated = self._step >= (self._max_episode_time_sec / self.dt)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
            1.0 / self.metadata["render_fps"]
        ):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action
        self._last_qpos = np.copy(self.data.qpos[7:])  # 保存当前关节位置供下一帧计算平滑度

        return observation, reward, terminated, truncated, info

    def _apply_domain_randomization(self):
        nv = self.model.nv
        nu = self.model.nu
        ngeom = self.model.ngeom
        nbody = self.model.nbody
        progress = self._curriculum_progress

        def _interp_range(base_range):
            low, high = base_range
            return (1.0 + (low - 1.0) * progress, 1.0 + (high - 1.0) * progress)

        if self._dr_enabled["friction"]:
            low, high = _interp_range(self._dr_scale_ranges["friction"])
            scales = self.np_random.uniform(low, high, size=ngeom)
            self.model.geom_friction[:,
                                     0] = self._base_geom_friction[:, 0] * scales

        if self._dr_enabled["mass"]:
            low, high = _interp_range(self._dr_scale_ranges["mass"])
            scales = self.np_random.uniform(low, high, size=nbody)
            self.model.body_mass[:] = self._base_body_mass[:] * scales

        if self._dr_enabled["damping"]:
            low, high = _interp_range(self._dr_scale_ranges["damping"])
            scales = self.np_random.uniform(low, high, size=nv)
            self.model.dof_damping[:] = self._base_dof_damping[:] * scales

        if self._dr_enabled["armature"]:
            low, high = _interp_range(self._dr_scale_ranges["armature"])
            scales = self.np_random.uniform(low, high, size=nv)
            self.model.dof_armature[:] = self._base_dof_armature[:] * scales

        if self._dr_enabled["frictionloss"]:
            low, high = _interp_range(self._dr_scale_ranges["frictionloss"])
            scales = self.np_random.uniform(low, high, size=nv)
            self.model.dof_frictionloss[:] = self._base_dof_frictionloss[:] * scales

        if self._dr_enabled["motor_kp"] and nu > 0:
            low, high = _interp_range(self._dr_scale_ranges["motor_kp"])
            scales = self.np_random.uniform(low, high, size=nu)
            for i in range(nu):
                self.model.actuator_gainprm[i,
                                            0] = self._base_actuator_gainprm[i, 0] * scales[i]
                self.model.actuator_biasprm[i,
                                            1] = self._base_actuator_biasprm[i, 1] * scales[i]

        if self._dr_enabled["gravity"]:
            gz_low, gz_high = self._dr_scale_ranges["gravity_z"]
            grav_offset = self.np_random.uniform(
                gz_low * progress, gz_high * progress)
            self.model.opt.gravity[2] = self._base_gravity[2] + grav_offset

        if self._dr_enabled["action_delay"]:
            delay_low = int(self._dr_action_delay_range[0] * progress)
            delay_high = max(delay_low, int(
                self._dr_action_delay_range[1] * progress))
            self._action_delay = (
                self.np_random.integers(delay_low, delay_high + 1)
                if delay_high > delay_low
                else delay_low
            )

        if self._dr_enabled["joint_bias"]:
            bias_low, bias_high = self._dr_scale_ranges["joint_bias"]
            max_bias = bias_high * progress
            self._joint_bias = self.np_random.uniform(
                -max_bias, max_bias, size=10
            )

        mujoco.mj_forward(self.model, self.data)

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z

        # 获取四元数并转换为欧拉角
        quat = self.data.qpos[3:7]  # w, x, y, z
        roll, pitch, _ = self.euler_from_quaternion(quat[0], quat[1], quat[2], quat[3])

        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= roll <= max_roll

        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= pitch <= max_pitch

        return is_healthy

    @property
    def projected_gravity(self):
        g_body = self._world_to_body_frame(self.model.opt.gravity)
        g_body_norm = np.linalg.norm(g_body)
        if g_body_norm > 0:
            return g_body / g_body_norm
        return g_body

    @property
    def feet_contact_forces(self):
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)

    ######### Positive Reward functions #########
    @property
    def linear_velocity_tracking_reward(self):
        body_linear_vel = self._world_to_body_frame(self.data.qvel[:3])
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - body_linear_vel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        body_angular_vel = self._world_to_body_frame(self.data.qvel[3:6])
        vel_sqr_error = np.square(
            self._desired_velocity[2] - body_angular_vel[2])
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def heading_tracking_reward(self):
        # TODO: qpos[3:7] are the quaternion values
        pass

    @property
    def energy_reward(self):
        # 获取关节扭矩和关节速度 (跳过 root 的 6 个自由度)
        tau = self.data.qfrc_actuator[6:]
        qvel = self.data.qvel[6:]
        
        # 分子：所有关节电机消耗的总功率 (取绝对值)
        power = np.sum(np.abs(tau) * np.abs(qvel))
        
        # 分母：基于线速度和角速度加权的广义运动距离
        body_linear_vel = self._world_to_body_frame(self.data.qvel[:3])
        body_angular_vel = self._world_to_body_frame(self.data.qvel[3:6])
        
        vx = np.abs(body_linear_vel[0])
        wz = np.abs(body_angular_vel[2])
        
        # 添加小值 0.1 防止分母为 0
        denominator = self.sigma_en_x * vx + self.sigma_en_z * wz + 0.1
        
        return np.exp(-power / denominator)

    @property
    def healthy_reward(self):
        return self.is_healthy

    ######### Negative Reward functions #########
    @property  # TODO: Not used
    def feet_contact_forces_cost(self):
        return np.sum(
            (self.feet_contact_forces - self._max_contact_force).clip(min=0.0)
        )

    @property
    def non_flat_base_cost(self):
        # Penalize the robot for not being flat on the ground
        return np.sum(np.square(self.projected_gravity[:2]))

    @property
    def collision_cost(self):
        # Penalize collisions on selected bodies
        return np.sum(
            1.0
            * (np.linalg.norm(self.data.cfrc_ext[self._cfrc_ext_contact_indices]) > 0.1)
        )

    @property
    def joint_limit_cost(self):
        # Penalize the robot for joints exceeding the soft control range
        out_of_range = (self._soft_joint_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        # All values are the motor torques (Turtle has 10 joints)
        return np.sum(np.square(self.data.qfrc_actuator))

    @property
    def vertical_velocity_cost(self):
        return np.square(self.data.qvel[2])

    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))

    def action_rate_cost(self, action):
        return np.sum(np.square(self._last_action - action))

    @property
    def joint_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[6:]))

    @property
    def acceleration_cost(self):
        return np.sum(np.square(self.data.qacc[6:]))

    @property
    def default_joint_position_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._default_joint_position))

    @property
    def smoothness_cost(self):
        """关节平滑度成本：当前关节角度与上一帧角度的 L2 距离"""
        return np.sum(np.square(self.data.qpos[7:] - self._last_qpos))

    @property
    def curriculum_factor(self):
        return max(0.0, 1.0 - self._curriculum_progress)

    @curriculum_factor.setter
    def curriculum_factor(self, value):
        self._curriculum_progress = np.clip(value, 0.0, 1.0)

    @property
    def curriculum_progress(self):
        return self._curriculum_progress

    @curriculum_progress.setter
    def curriculum_progress(self, value):
        self._curriculum_progress = np.clip(value, 0.0, 1.0)

    def _calc_reward(self, action):
        # TODO: Add debug mode with custom Tensorboard calls for individual reward
        #   functions to get a better sense of the contribution of each reward function
        # TODO: Cost for thigh or calf contact with the ground

        # 能量奖励权重随课程进度从 0 逐渐增加到目标值，让智能体先学会基本步态再优化能量效率
        # progress < 0.3: alpha_en = 0 (完全关闭能量奖励)
        # 0.3 <= progress <= 1.0: alpha_en 线性增长到目标值 1.0
        progress = self._curriculum_progress
        if progress < 0.3:
            dynamic_alpha_en = 0.0
        else:
            dynamic_alpha_en = self.alpha_en * min(1.0, (progress - 0.3) / 0.7)

        # 1. Motion Rewards (R_motion)
        r_motion = (
            self.linear_velocity_tracking_reward * self.reward_weights["linear_vel_tracking"]
            + self.alpha_ang * self.angular_velocity_tracking_reward * self.reward_weights["angular_vel_tracking"]
        )
        
        # 2. Energy Reward (R_en) - 使用动态权重
        r_en = self.energy_reward
        
        # 3. Auxiliary Costs (R_aux)
        ctrl_cost = self.torque_cost * self.cost_weights["torque"]
        action_rate_cost = (
            self.action_rate_cost(action) * self.cost_weights["action_rate"]
        )
        vertical_vel_cost = (
            self.vertical_velocity_cost * self.cost_weights["vertical_vel"]
        )
        xy_angular_vel_cost = (
            self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
        )
        joint_limit_cost = self.joint_limit_cost * \
            self.cost_weights["joint_limit"]
        joint_velocity_cost = (
            self.joint_velocity_cost * self.cost_weights["joint_velocity"]
        )
        joint_acceleration_cost = (
            self.acceleration_cost * self.cost_weights["joint_acceleration"]
        )
        orientation_cost = self.non_flat_base_cost * \
            self.cost_weights["orientation"]
        collision_cost = self.collision_cost * self.cost_weights["collision"]
        default_joint_position_cost = (
            self.default_joint_position_cost
            * self.cost_weights["default_joint_position"]
        )
        
        smoothness_cost = self.smoothness_cost
        r_aux = (
            ctrl_cost + action_rate_cost + vertical_vel_cost + xy_angular_vel_cost +
            joint_limit_cost + joint_velocity_cost + joint_acceleration_cost +
            orientation_cost + default_joint_position_cost + collision_cost 
            # + smoothness_cost
        )
        
        # 4. Total Reward: R = (R_motion + dynamic_alpha_en * R_en) * exp(-clip(R_aux, 0, 2.0))
        reward = (r_motion + dynamic_alpha_en * r_en) * np.exp(-np.clip(r_aux, 0, 2.0))  # clip r_aux 防止初期成本过高导致奖励被归零
        
        reward_info = {
            "r_motion": r_motion,
            "r_en": r_en,
            "r_aux": r_aux,
            "reward_total": reward,
        }

        return reward, reward_info

    def _get_obs(self):
        dofs_position = self.data.qpos[7:].flatten(
        ) - self.model.key_qpos[0, 7:]

        velocity = self.data.qvel.flatten()
        base_linear_velocity = self._world_to_body_frame(velocity[:3].copy())
        base_angular_velocity = self._world_to_body_frame(velocity[3:6].copy())
        dofs_velocity = velocity[6:].copy()

        if self._dr_enabled["sensor_noise"]:
            nl = self._dr_noise_levels
            progress = self._curriculum_progress
            base_linear_velocity += self.np_random.normal(
                0, nl["linear_velocity"] * progress, size=3
            )
            base_angular_velocity += self.np_random.normal(
                0, nl["angular_velocity"] * progress, size=3
            )
            dofs_position += self.np_random.normal(
                0, nl["dofs_position"] * progress, size=10
            )
            dofs_velocity += self.np_random.normal(
                0, nl["dofs_velocity"] * progress, size=10
            )

        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.projected_gravity

        if self._dr_enabled["sensor_noise"]:
            projected_gravity += self.np_random.normal(
                0, self._dr_noise_levels["projected_gravity"] * progress, size=3
            )
            projected_gravity_norm = np.linalg.norm(projected_gravity)
            if projected_gravity_norm > 0:
                projected_gravity /= projected_gravity_norm

        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],
                base_angular_velocity * self._obs_scale["angular_velocity"],
                projected_gravity,
                # 分开缩放线速度和角速度目标值
                desired_vel[:2] * self._obs_scale["linear_velocity"],
                desired_vel[2:3] * self._obs_scale["angular_velocity"],
                dofs_position * self._obs_scale["dofs_position"],
                dofs_velocity * self._obs_scale["dofs_velocity"],
                last_action,
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        return curr_obs

    def reset_model(self):
        self._apply_domain_randomization()

        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[
            0
        ] + self._reset_noise_scale * self.np_random.standard_normal(
            *self.data.ctrl.shape
        )

        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(10)
        self._action_buffer = np.zeros_like(self._action_buffer)
        self._last_render_time = -1.0

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _sample_desired_vel(self):
        # 固定目标速度，移除 progress 乘数以提供稳定且足够大的期望速度
        # vx = 0 (左右保持静止), vy = -0.5 (向前), wz = 0 (无旋转)
        desired_vel = np.array([0.0, -0.5, 0.0])
        return desired_vel

    def _world_to_body_frame(self, world_vec):
        quat = self.data.qpos[3:7]
        w, x, y, z = quat
        q_vec = np.array([-x, -y, -z])
        t = 2.0 * np.cross(q_vec, world_vec)
        return world_vec + w * t + np.cross(q_vec, t)

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians
