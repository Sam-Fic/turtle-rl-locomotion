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

    def __init__(self, ctrl_type="position", **kwargs):
        self._curriculum_base = 0.3
        self._curriculum_progress = 0.0
        self._step_count = 0

        self._dr_enabled = {
            "friction": True,
            "mass": True,
            "damping": True,
            "armature": True,
            "frictionloss": True,
            "motor_kp": True,
            "gravity": True,
            "sensor_noise": True,
            "action_delay": True,
            "joint_bias": True,
        }
        self._dr_scale_ranges = {
            "friction": (0.5, 1.8),
            "mass": (0.7, 1.3),
            "damping": (0.3, 3.0),
            "armature": (0.3, 3.0),
            "frictionloss": (0.0, 2.0),
            "motor_kp": (0.6, 1.4),
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
            "linear_vel_tracking": 2.0,  # Was 1.0
            "angular_vel_tracking": 1.0,
            "healthy": 0.0,  # was 0.05
            "feet_airtime": 1.0,
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 2.0,  # Was 1.0
            "xy_angular_vel": 0.05,  # Was 0.05
            "action_rate": 0.01,
            "joint_limit": 10.0,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7,
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.1
        }

        self._gravity_vector = np.array(self.model.opt.gravity)
        self._default_joint_position = np.array(self.model.key_ctrl[0])

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
        self._tracking_velocity_sigma = 0.25

        # Metrics used to determine if the episode should be terminated
        self._healthy_z_range = (0.08, 0.40)
        self._healthy_pitch_range = (-np.deg2rad(10), np.deg2rad(10))
        self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
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

        self._reset_noise_scale = 0.2

        # Action: 10 torque values (Turtle has 10 joints)
        self._last_action = np.zeros(10)

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
        self._step_count += 1
        self._curriculum_progress = np.clip(
            self._step_count / 10_000_000, 0.0, 1.0
        )

        self._action_buffer = np.roll(self._action_buffer, shift=1, axis=0)
        self._action_buffer[0] = action
        delayed_action = self._action_buffer[self._action_delay]
        biased_action = np.clip(
            delayed_action + self._joint_bias,
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
            self.model.geom_friction[:, 0] *= scales

        if self._dr_enabled["mass"]:
            low, high = _interp_range(self._dr_scale_ranges["mass"])
            scales = self.np_random.uniform(low, high, size=nbody)
            self.model.body_mass[:] *= scales

        if self._dr_enabled["damping"]:
            low, high = _interp_range(self._dr_scale_ranges["damping"])
            scales = self.np_random.uniform(low, high, size=nv)
            self.model.dof_damping[:] *= scales

        if self._dr_enabled["armature"]:
            low, high = _interp_range(self._dr_scale_ranges["armature"])
            scales = self.np_random.uniform(low, high, size=nv)
            self.model.dof_armature[:] *= scales

        if self._dr_enabled["frictionloss"]:
            low, high = _interp_range(self._dr_scale_ranges["frictionloss"])
            scales = self.np_random.uniform(low, high, size=nv)
            self.model.dof_frictionloss[:] *= scales

        if self._dr_enabled["motor_kp"] and nu > 0:
            low, high = _interp_range(self._dr_scale_ranges["motor_kp"])
            scales = self.np_random.uniform(low, high, size=nu)
            for i in range(nu):
                self.model.actuator_gainprm[i, 0] *= scales[i]
                self.model.actuator_biasprm[i, 1] *= scales[i]

        if self._dr_enabled["gravity"]:
            gz_low, gz_high = self._dr_scale_ranges["gravity_z"]
            grav_offset = self.np_random.uniform(
                gz_low * progress, gz_high * progress)
            self.model.opt.gravity[2] += grav_offset

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

        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= state[4] <= max_roll

        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= state[5] <= max_pitch

        return is_healthy

    @property
    def projected_gravity(self):
        w, x, y, z = self.data.qpos[3:7]
        euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z))
        projected_gravity_not_normalized = (
            np.dot(self._gravity_vector, euler_orientation) * euler_orientation
        )
        if np.linalg.norm(projected_gravity_not_normalized) == 0:
            return projected_gravity_not_normalized
        else:
            return projected_gravity_not_normalized / np.linalg.norm(
                projected_gravity_not_normalized
            )

    @property
    def feet_contact_forces(self):
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)

    ######### Positive Reward functions #########
    @property
    def linear_velocity_tracking_reward(self):
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - self.data.qvel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        vel_sqr_error = np.square(
            self._desired_velocity[2] - self.data.qvel[5])
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def heading_tracking_reward(self):
        # TODO: qpos[3:7] are the quaternion values
        pass

    @property
    def feet_air_time_reward(self):
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 1.0) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return air_time_reward

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
        return np.sum(np.square(self.data.qpos[7:] - self._last_action))

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

        # Positive Rewards
        linear_vel_tracking_reward = (
            self.linear_velocity_tracking_reward
            * self.reward_weights["linear_vel_tracking"]
        )
        angular_vel_tracking_reward = (
            self.angular_velocity_tracking_reward
            * self.reward_weights["angular_vel_tracking"]
        )
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = (
            self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        )
        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
        )

        # Negative Costs
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
        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_acceleration_cost
            + orientation_cost
            + default_joint_position_cost
        )

        reward = max(0.0, rewards - costs)
        # reward = rewards - self.curriculum_factor * costs
        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        dofs_position = self.data.qpos[7:].flatten(
        ) - self.model.key_qpos[0, 7:]

        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3].copy()
        base_angular_velocity = velocity[3:6].copy()
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
                desired_vel * self._obs_scale["linear_velocity"],
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
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
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
        progress = self._curriculum_progress
        vel_min = np.array([-0.2 * progress, -0.5, -0.3 * progress])
        vel_max = np.array(
            [0.2 * progress, -0.1 - 0.4 * progress, 0.3 * progress])
        desired_vel = self.np_random.uniform(low=vel_min, high=vel_max)
        return desired_vel

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
