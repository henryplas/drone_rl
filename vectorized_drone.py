import numpy as np
import gym
from gym import spaces
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# =============================================================================
# Vectorized Drone Dynamics Implementation with Motor Forces as Inputs
# =============================================================================
class VectorizedDroneEnv:
    def __init__(self, batch_size=10, dt=0.02):
        # Batch size: number of parallel environments
        self.batch_size = batch_size
        
        # Drone physical parameters
        self.mass = 1.0         # kg
        self.g = 9.81           # m/s^2
        # Diagonal inertia matrix: [Ixx, Iyy, Izz]
        self.I = np.array([0.005, 0.005, 0.01])
        self.dt = dt            # time step

        # Geometry for the X configuration (arm length)
        self.arm_length = 0.5
        # Yaw moment coefficient mapping motor force difference to yaw torque.
        self.k_yaw = 0.01

        # Target waypoint (hover at z = 10)
        self.target = np.array([0.0, 0.0, 10.0])
        
        # Maximum steps per episode
        self.max_steps = 1000

        # Initialize states (each of shape (batch_size, 3)).
        self.reset()

    def reset(self):
        """
        Resets the state of all drones.
        Each state has:
          - pos: [x, y, z]
          - vel: [vx, vy, vz]
          - euler: [roll, pitch, yaw] (rad)
          - omega: [p, q, r] (rad/s)
        Returns:
          obs: (batch_size, 12) observation array.
        """
        # For example, all drones starting near the origin with a small offset.
        self.pos = np.tile(np.array([0.1, 0.1, 0.1]), (self.batch_size, 1))
        self.vel = np.zeros((self.batch_size, 3))
        self.euler = np.zeros((self.batch_size, 3))
        self.omega = np.zeros((self.batch_size, 3))
        
        # Keep track of the number of steps (synchronized for all in the batch)
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        """Concatenates state variables into a (batch_size, 12) array."""
        return np.concatenate([self.pos, self.vel, self.euler, self.omega], axis=1).astype(np.float32)
    
    def _rotation_matrix(self, euler):
        """
        Computes the rotation matrix for each drone from body to inertial frame using
        ZYX Euler angles.
        
        Parameters:
          euler: (batch_size, 3) array with [roll, pitch, yaw] for each drone.
          
        Returns:
          R: (batch_size, 3, 3) array where each R[i] is a rotation matrix.
        """
        phi = euler[:, 0]
        theta = euler[:, 1]
        psi = euler[:, 2]
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        # Build rotation matrices for each environment.
        R = np.empty((self.batch_size, 3, 3))
        R[:, 0, 0] = c_psi * c_theta
        R[:, 0, 1] = c_psi * s_theta * s_phi - s_psi * c_phi
        R[:, 0, 2] = c_psi * s_theta * c_phi + s_psi * s_phi

        R[:, 1, 0] = s_psi * c_theta
        R[:, 1, 1] = s_psi * s_theta * s_phi + c_psi * c_phi
        R[:, 1, 2] = s_psi * s_theta * c_phi - c_psi * s_phi

        R[:, 2, 0] = -s_theta
        R[:, 2, 1] = c_theta * s_phi
        R[:, 2, 2] = c_theta * c_phi
        
        return R

    def _euler_angle_rates(self, euler, omega):
        """
        Computes the derivative of Euler angles given each drone's angular velocity.
        
        Parameters:
          euler: (batch_size, 3) array (roll, pitch, yaw)
          omega: (batch_size, 3) angular velocity (p, q, r)
          
        Returns:
          euler_dot: (batch_size, 3) time derivative of Euler angles.
        """
        phi = euler[:, 0]
        theta = euler[:, 1]
        
        # Precompute values for T matrix elements.
        tan_theta = np.tan(theta)
        sec_theta = 1 / np.cos(theta)
        
        # T has shape (batch_size, 3, 3)
        T = np.empty((self.batch_size, 3, 3))
        T[:, 0, 0] = 1.0
        T[:, 0, 1] = np.sin(phi) * tan_theta
        T[:, 0, 2] = np.cos(phi) * tan_theta
        T[:, 1, 0] = 0.0
        T[:, 1, 1] = np.cos(phi)
        T[:, 1, 2] = -np.sin(phi)
        T[:, 2, 0] = 0.0
        T[:, 2, 1] = np.sin(phi) * sec_theta
        T[:, 2, 2] = np.cos(phi) * sec_theta

        # Compute euler_dot for each environment.
        # Using np.einsum to perform batch matrix multiplication: (batch,3,3) * (batch,3) -> (batch,3)
        euler_dot = np.einsum('bij,bj->bi', T, omega)
        return euler_dot

    def step(self, action):
        """
        Updates the state of all drones based on the input motor forces.
        
        Parameters:
          action: (batch_size, 4) array, each row represents:
                  [F1, F2, F3, F4] motor forces for that drone.
                  (The forces are assumed to act along the drone's body-z axis.)
        
        Returns:
          obs: (batch_size, 12) observation array.
          reward: (batch_size,) reward computed per drone (negative distance to target, with bonus near target).
          done: (batch_size,) boolean array where True indicates termination for that drone.
          info: an empty dict.
        """
        # Motor forces for each drone (shape: (batch_size, 4))
        motor_forces = action

        # Compute net thrust and torque for each drone.
        thrust = np.sum(motor_forces, axis=1)  # shape: (batch_size,)
        L = self.arm_length
        factor = L / np.sqrt(2)

        # Roll torque: tau_phi = factor * (F1 + F2 - F3 - F4)
        tau_phi = factor * (motor_forces[:, 0] + motor_forces[:, 1] -
                            motor_forces[:, 2] - motor_forces[:, 3])
        # Pitch torque: tau_theta = factor * (-F1 + F2 + F3 - F4)
        tau_theta = factor * (-motor_forces[:, 0] + motor_forces[:, 1] +
                              motor_forces[:, 2] - motor_forces[:, 3])
        # Yaw torque: tau_psi = k_yaw * (F1 - F2 + F3 - F4)
        tau_psi = self.k_yaw * (motor_forces[:, 0] - motor_forces[:, 1] +
                                motor_forces[:, 2] - motor_forces[:, 3])

        # Compute rotation matrix for all drones: (batch_size, 3,3)
        R = self._rotation_matrix(self.euler)
        
        # Compute acceleration.
        # Thrust vector in body frame, for each environment:
        # For each row: [0, 0, thrust].
        thrust_body = np.zeros((self.batch_size, 3))
        thrust_body[:, 2] = thrust

        # Convert thrust vector to inertial frame: use batch matrix multiplication.
        # Using np.einsum to compute dot product: (batch,3,3) and (batch,3) -> (batch,3)
        thrust_inertial = np.einsum('bij,bj->bi', R, thrust_body)
        # Include gravity acceleration.
        accel = np.tile(np.array([0, 0, -self.g]), (self.batch_size, 1)) + (thrust_inertial / self.mass)
        
        # Update linear state (Euler integration)
        self.vel += accel * self.dt
        self.pos += self.vel * self.dt

        # Update angular state (Euler angles)
        euler_dot = self._euler_angle_rates(self.euler, self.omega)
        self.euler += euler_dot * self.dt

        # Angular acceleration calculation:
        omega_dot = np.empty_like(self.omega)
        # For each axis, using vectorized math:
        omega_dot[:, 0] = (tau_phi - (self.I[1] - self.I[2]) * self.omega[:, 1] * self.omega[:, 2]) / self.I[0]
        omega_dot[:, 1] = (tau_theta - (self.I[2] - self.I[0]) * self.omega[:, 0] * self.omega[:, 2]) / self.I[1]
        omega_dot[:, 2] = (tau_psi - (self.I[0] - self.I[1]) * self.omega[:, 0] * self.omega[:, 1]) / self.I[2]
        self.omega += omega_dot * self.dt

        # Update the step count.
        self.current_step += 1

        # Compute reward as the negative Euclidean distance (scaled) from target.
        # For each drone, use np.linalg.norm along axis 1.
        distance = np.linalg.norm(self.pos - self.target, axis=1)
        reward = -0.01 * distance
        # Add bonus if a drone is very close to the target.
        reward[distance < 1] += 1

        # Determine termination conditions for each drone.
        # Done if z < 0 (crash) or if drone drifts too far (norm > 50), or if max steps reached.
        done = (self.pos[:, 2] < 0) | (np.linalg.norm(self.pos, axis=1) > 50)
        if self.current_step >= self.max_steps:
            done = np.ones(self.batch_size, dtype=bool)

        info = {}
        return self._get_obs(), reward, done, info

    def render(self, ax=None):
        """
        Renders the state of all drones in the batch onto a 3D plot.
        This simplistic renderer plots the target waypoint and the position of each drone.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.clear()
        
        # Plot target waypoint.
        ax.scatter(self.target[0], self.target[1], self.target[2],
                   color='green', s=50, label='Target')
        # Plot drone centers.
        ax.scatter(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
                   color='red', s=20, label='Drone Centers')
        
        # Set plot limits.
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([0, 20])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.draw()
        plt.pause(0.001)


# =============================================================================
# Optional Gym-Compatible Wrapper for the Vectorized Environment
# (Note: Gym’s typical API is for single-environment instances. For parallelism,
# you may also consider using vectorized environments from stable-baselines3 or Ray.)
# =============================================================================
class VectorizedDroneGymEnv(VectorizedDroneEnv, gym.Env):
    def __init__(self, batch_size=10, dt=0.02):
        VectorizedDroneEnv.__init__(self, batch_size, dt)
        gym.Env.__init__(self)
        # Observation: each env has 12 state variables.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.batch_size, 12), dtype=np.float32)
        # Action: each env uses 4 motor forces.
        # Here we assume each motor's force is between 0 and 3*mass*g/4.
        motor_max = 3 * self.mass * self.g / 4.0
        self.action_space = spaces.Box(low=0, high=motor_max, shape=(self.batch_size, 4), dtype=np.float32)

    def step(self, action):
        return super().step(action)

    def reset(self):
        return super().reset()

    def render(self, mode="human", close=False):
        return super().render()


# =============================================================================
# Example usage of the vectorized environment:
# =============================================================================
if __name__ == "__main__":
    # Create a vectorized environment with, for example, 5 drones in parallel.
    env = VectorizedDroneEnv(batch_size=5)
    obs = env.reset()
    
    # Setup a figure for visualization.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for _ in range(300):
        # For a hovering condition, each motor should provide roughly (mass*g/4).
        hover_force_per_motor = (env.mass * env.g) / 4.0
        # For example purposes, apply a constant force slightly above hover.
        action = np.full((env.batch_size, 4), hover_force_per_motor * 2)
        obs, reward, done, info = env.step(action)
        
        env.render(ax=ax)
        plt.pause(0.02)
        
        # Optionally, you can check if all environments are done.
        if np.all(done):
            print("All environments terminated!")
            break
