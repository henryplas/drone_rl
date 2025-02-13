import numpy as np
import gym
from gym import spaces
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Drone dynamics implementation
# ---------------------------------------------------------------------------
class DroneEnv:
    def __init__(self, dt=0.02):
        # Drone physical parameters
        self.mass = 1.0         # kg
        self.g = 9.81           # m/s^2
        # Diagonal inertia matrix: [Ixx, Iyy, Izz]
        self.I = np.array([0.005, 0.005, 0.01])
        self.dt = dt            # time step

        # For drawing: arm length for the X shape (distance from center to each motor)
        self.arm_length = 0.5

        # Desired target waypoint (for example, hover at z = 10)
        self.target = np.array([0.0, 0.0, 10.0])
        
        # Initialize state
        self.reset()

    def reset(self):
        """
        Resets the drone state.
        State consists of:
          - pos: [x, y, z] position in inertial frame
          - vel: [vx, vy, vz] linear velocity
          - euler: [roll, pitch, yaw] angles (rad)
          - omega: [p, q, r] angular velocity (rad/s)
        """
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.omega = np.zeros(3)
        return self._get_obs()

    def _get_obs(self):
        """Returns the full state as a 1D numpy array."""
        return np.concatenate([self.pos, self.vel, self.euler, self.omega]).astype(np.float32)

    def step(self, action):
        """
        Updates the state given an action.
        
        Parameters:
          action: np.array with 4 elements:
                  [thrust, tau_phi, tau_theta, tau_psi]
                  
        Returns:
          obs: next state observation
          reward: negative Euclidean distance from target
          done: boolean indicating termination (if drone crashes or drifts too far)
          info: additional info (empty dict)
        """
        thrust, tau_phi, tau_theta, tau_psi = action

        # Compute rotation matrix from body to inertial frame.
        R = self._rotation_matrix(self.euler)
        
        # Linear dynamics: gravity plus thrust (applied along the drone's body z-axis)
        thrust_body = np.array([0, 0, thrust])
        accel = np.array([0, 0, -self.g]) + (R @ thrust_body) / self.mass
        
        # Euler integration for linear state.
        self.vel += accel * self.dt
        self.pos += self.vel * self.dt

        # Update angular state:
        euler_dot = self._euler_angle_rates(self.euler, self.omega)
        self.euler += euler_dot * self.dt
        
        # Angular dynamics (assuming diagonal inertia)
        omega_dot = np.zeros(3)
        omega_dot[0] = (tau_phi - (self.I[1] - self.I[2]) * self.omega[1] * self.omega[2]) / self.I[0]
        omega_dot[1] = (tau_theta - (self.I[2] - self.I[0]) * self.omega[0] * self.omega[2]) / self.I[1]
        omega_dot[2] = (tau_psi - (self.I[0] - self.I[1]) * self.omega[0] * self.omega[1]) / self.I[2]
        self.omega += omega_dot * self.dt

        # Reward: negative distance from the target
        reward = -np.linalg.norm(self.pos - self.target)
        # Terminate if the drone "crashes" (z < 0) or drifts too far from the origin.
        done = self.pos[2] < 0 or np.linalg.norm(self.pos) > 50
        info = {}
        return self._get_obs(), reward, done, info

    def _rotation_matrix(self, euler):
        """
        Returns the rotation matrix (body -> inertial) using ZYX Euler angles.
        """
        phi, theta, psi = euler  # roll, pitch, yaw
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        R = np.array([
            [c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
            [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi],
            [-s_theta,        c_theta * s_phi,                          c_theta * c_phi]
        ])
        return R

    def _euler_angle_rates(self, euler, omega):
        """
        Computes the time derivative of Euler angles given the body angular velocity.
        """
        phi, theta, _ = euler
        T = np.array([
            [1, np.sin(phi)*np.tan(theta),  np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi),               -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta),   np.cos(phi)/np.cos(theta)]
        ])
        return T @ omega

    def render(self, ax=None):
        """
        Renders a 3D visualization of the drone.
        The drone is drawn as an “X” shape formed by two lines connecting opposite motors.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        ax.clear()
        # Plot the target waypoint.
        ax.scatter(self.target[0], self.target[1], self.target[2],
                   color='green', s=50, label='Target')
        
        # Compute motor positions in the drone's body frame.
        # Motors are arranged at 45°, 135°, 225°, and 315°.
        arm = self.arm_length
        motor_offsets = np.array([
            [ arm/np.sqrt(2),  arm/np.sqrt(2), 0],
            [-arm/np.sqrt(2),  arm/np.sqrt(2), 0],
            [-arm/np.sqrt(2), -arm/np.sqrt(2), 0],
            [ arm/np.sqrt(2), -arm/np.sqrt(2), 0]
        ])
        # Rotate motor positions to the inertial frame.
        R = self._rotation_matrix(self.euler)
        motors_inertial = self.pos + (R @ motor_offsets.T).T  # shape (4,3)
        
        # Draw two lines forming an X: one between motor 0 and 2, and one between motor 1 and 3.
        ax.plot([motors_inertial[0, 0], motors_inertial[2, 0]],
                [motors_inertial[0, 1], motors_inertial[2, 1]],
                [motors_inertial[0, 2], motors_inertial[2, 2]],
                color='purple', linewidth=2)
        ax.plot([motors_inertial[1, 0], motors_inertial[3, 0]],
                [motors_inertial[1, 1], motors_inertial[3, 1]],
                [motors_inertial[1, 2], motors_inertial[3, 2]],
                color='purple', linewidth=2)
        
        # Optionally, plot the drone center and motor positions.
        ax.scatter(self.pos[0], self.pos[1], self.pos[2],
                   color='red', s=20, label='Center')
        ax.scatter(motors_inertial[:, 0], motors_inertial[:, 1], motors_inertial[:, 2],
                   color='blue', s=20, label='Motors')
        
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([0, 20])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.draw()
        plt.pause(0.001)

# ---------------------------------------------------------------------------
# Gym-compatible environment wrapping the DroneEnv
# ---------------------------------------------------------------------------
class DroneGymEnv(DroneEnv, gym.Env):
    def __init__(self, dt=0.02):
        DroneEnv.__init__(self, dt)
        gym.Env.__init__(self)
        # Observation: [pos(3), vel(3), euler(3), omega(3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        # Action: [thrust, tau_phi, tau_theta, tau_psi]
        action_low = np.array([0, -1, -1, -1], dtype=np.float32)
        action_high = np.array([2 * self.mass * self.g, 1, 1, 1], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def step(self, action):
        return super().step(action)

    def reset(self):
        return super().reset()

    def render(self, mode="human", close=False):
        return super().render()


# Example usage:
if __name__ == "__main__":
    env = DroneEnv()
    obs = env.reset()
    
    # Create a figure for visualization.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for _ in range(300):
        # Example action: constant thrust to counteract gravity (hover).
        action = np.array([env.mass * env.g * 2, 0, 0, 0])
        obs, reward, done, info = env.step(action)
        
        env.render(ax=ax)
        plt.pause(0.02)
        
        if done:
            print("Episode ended!")
            break
