import numpy as np
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


# ---------------------------------------------------------------------------
# Drone dynamics implementation with motor force inputs
# ---------------------------------------------------------------------------
class DroneEnv:
    def __init__(self, dt=0.02):
        self._fig = None
        self._writer = None
    
        self.ep_num = 0
        self.total_steps = 1
        # Drone physical parameters
        self.mass = 1.0         # kg
        self.g = 9.81           # m/s^2
        # Diagonal inertia matrix: [Ixx, Iyy, Izz]
        self.I = np.array([0.005, 0.005, 0.01])
        self.dt = dt            # time step


        self.current_step = 0
        
        self.add = 0


        self.eps = 0.0
        
        # For drawing: arm length for the X shape (distance from center to each motor)
        self.arm_length = 0.5
        
        # Yaw moment coefficient: maps differential motor forces to yaw torque.
        # Adjust this constant as needed.
        self.k_yaw = 0.01

         # Maximum steps per episode (added for longer episodes)
        self.max_steps = 200

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
        self.pos = np.array([np.random.rand()-0.5, np.random.rand()-0.5, 1.0])
        self.vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.omega = np.zeros(3)
        self.ep_num += 1
        # if self.ep_num % 2000 == 0 and eps < 3:
        #     eps+=0.1
        # self.amt = 20000 // self.current_step

        self.current_step = 0

        if self.ep_num % 2000 == 0:
            # self.add += 0.1
            self.eps += 0.1


        self.target = np.array([self.eps * np.random.rand(), self.eps * np.random.rand(), self.eps * np.random.rand() + 1.0 + self.add ])

        return self._get_obs()

    def _get_obs(self):
        """Returns the full state as a 1D numpy array."""
        return np.concatenate([self.pos, self.vel, self.euler, self.omega, self.target-self.pos]).astype(np.float32)

    def step(self, action):
        """
        Updates the state given an action.
        
        Parameters:
          action: np.array with 4 elements representing the forces (in Newtons) produced by each motor:
                  [F1, F2, F3, F4]
          Note: It is assumed that motor forces act along the drone's body z-axis.
                The mapping to total thrust and body torques is given by:
                   Total thrust: F_total = F1 + F2 + F3 + F4
                   Roll torque (tau_phi):    L/sqrt2 * (F1 + F2 - F3 - F4)
                   Pitch torque (tau_theta): L/sqrt2 * (-F1 + F2 + F3 - F4)
                   Yaw torque (tau_psi):     k_yaw * (F1 - F2 + F3 - F4)
        
        Returns:
          obs: next state observation
          reward: negative Euclidean distance from target (with extra bonus when near the target)
          done: boolean indicating termination (if drone crashes or drifts too far)
          info: additional info (empty dict)
        """
        self.total_steps += 1
        # Interpret action as individual motor forces.
        motor_forces = action  # [F1, F2, F3, F4]

        # Compute net thrust and torques
        thrust = np.sum(motor_forces)
        # Using the motor positions defined in render() (X configuration):
        # Motor offsets in body frame:
        # Motor 1:  [L/sqrt2,  L/sqrt2, 0]
        # Motor 2:  [-L/sqrt2, L/sqrt2, 0]
        # Motor 3:  [-L/sqrt2, -L/sqrt2, 0]
        # Motor 4:  [L/sqrt2,  -L/sqrt2, 0]
        L = self.arm_length
        factor = L / np.sqrt(2)
        tau_phi = factor * (motor_forces[0] + motor_forces[1] - motor_forces[2] - motor_forces[3])
        tau_theta = factor * (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3])
        tau_psi = self.k_yaw * (motor_forces[0] - motor_forces[1] + motor_forces[2] - motor_forces[3])

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

        # Reward: negative distance from the target (plus bonus if near target)
        reward = 0.01 * -np.linalg.norm(self.pos - self.target)  
        # if np.linalg.norm(self.pos - self.target) < 1.0:
        #     reward += .1
        # if np.linalg.norm(self.pos - self.target) < .25:
        #     reward += .1
        if np.linalg.norm(self.pos - self.target) < .05:
            reward += 1
        # if self.current_step < 10 and np.all(motor_forces>0):
        #     reward += 0.01

        
        # Terminate if the drone "crashes" (z < 0) or drifts too far from the origin.
        done = self.pos[2] < 0 or np.linalg.norm(self.pos) > 50
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
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


    def start_record(self, filename='drone_run.mp4', dpi=200, fps=20, bitrate=-1):
        """
        Call before your loop to start video recording.
        Requires ffmpeg on your PATH.
        """
        if self._fig is None:
            self._fig = plt.figure()
        self._writer = PillowWriter(fps=fps)
        self._writer.setup(self._fig, filename, dpi)

    def stop_record(self):
        """Call after your loop to finish & save the video."""
        if self._writer:
            self._writer.finish()
            self._writer = None

    def render(self, ax=None):
        """
        Renders a 3D visualization of the drone.
        If recording, each frame is grabbed into the video.
        """
        # create figure & ax if needed
        if ax is None:
            if self._fig is None:
                self._fig = plt.figure()
            ax = self._fig.add_subplot(111, projection='3d')

        ax.clear()
        # — your existing plotting code —
        ax.scatter(self.target[0], self.target[1], self.target[2],
                   color='green', s=50, label='Target')
        arm = self.arm_length
        motor_offsets = np.array([
            [ arm/np.sqrt(2),  arm/np.sqrt(2), 0],
            [-arm/np.sqrt(2),  arm/np.sqrt(2), 0],
            [-arm/np.sqrt(2), -arm/np.sqrt(2), 0],
            [ arm/np.sqrt(2), -arm/np.sqrt(2), 0]
        ])
        R = self._rotation_matrix(self.euler)
        motors = self.pos + (R @ motor_offsets.T).T
        ax.plot([motors[0,0], motors[2,0]],
                [motors[0,1], motors[2,1]],
                [motors[0,2], motors[2,2]], color='purple', lw=2)
        ax.plot([motors[1,0], motors[3,0]],
                [motors[1,1], motors[3,1]],
                [motors[1,2], motors[3,2]], color='purple', lw=2)
        ax.scatter(self.pos[0], self.pos[1], self.pos[2],
                   color='red', s=20, label='Center')
        ax.scatter(motors[:,0], motors[:,1], motors[:,2],
                   color='blue', s=20, label='Motors')
        ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_zlim(0,5)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        plt.draw()
        if self._writer:
            # grab this frame for the video
            self._writer.grab_frame()
        else:
            # normal live-render
            plt.pause(0.001)

             
# ---------------------------------------------------------------------------
# Gym-compatible environment wrapping the DroneEnv
# ---------------------------------------------------------------------------
class DroneGymEnv(DroneEnv, gym.Env):
    def __init__(self, dt=0.02):
        DroneEnv.__init__(self, dt)
        gym.Env.__init__(self)
        # Observation: [pos(3), vel(3), euler(3), omega(3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        # Action: now each element is the force from each motor.
        # For this example, we assume each motor's force is between 0 and 3*mass*g/4,
        # so that the maximum total force is 3*mass*g.
        motor_max = 3 * self.mass * self.g / 4.0
        self.action_space = spaces.Box(low=0, high=motor_max, shape=(4,), dtype=np.float32)

    def step(self, action):

        return super().step(action)

    def reset(self):
        return super().reset()

    def render(self, mode="human", close=False):
        return super().render()


# ---------------------------------------------------------------------------
# Example usage:
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = DroneEnv()
    obs = env.reset()
    
    # Create a figure for visualization.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for _ in range(300):
        # For a hovering condition the total thrust required is mass * g.
        # With four motors, each motor should provide (mass*g)/4 for hover.
        # Here we apply a constant force slightly above hover conditions (e.g. 2 times hover force distributed across motors).
        hover_force_per_motor = (env.mass * env.g) / 4.0
        action = np.full(4, hover_force_per_motor * 2)
        obs, reward, done, info = env.step(action)
        
        env.render(ax=ax)
        plt.pause(0.02)
        
        if done:
            print("Episode ended!")
            break
