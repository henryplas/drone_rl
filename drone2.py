import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation
import json
import os

# ---- Configuration (Add max_torque) ----
CONFIG_FILE = 'drone_config.json'
DEFAULT_CONFIG = {
    'initial_pos': [0.0, 0.0, 1.0],
    'initial_vel': [0.0, 0.0, 0.0],
    'initial_att': [0.0, 0.0, 0.0],
    'initial_omega': [0.0, 0.0, 0.0],
    'mass': 1.0,
    'inertia_diag': [0.005, 0.005, 0.01],
    'g': 9.81,
    'max_thrust_per_prop': 5.0,
    'num_propellers': 4,
    'max_torque': [0.5, 0.5, 0.2], # ADDED: Max controllable torque [Tx, Ty, Tz] (N*m) - TUNE THIS!
    'controller_gains': {
        # --- GAINS BELOW LIKELY NEED TUNING ---
        'Kp_pos_diag': [2.0, 2.0, 5.0],  # Try reducing these first (e.g., [1.0, 1.0, 2.0])
        'Kd_pos_diag': [2.5, 2.5, 4.0],  # Try reducing these (e.g., [1.5, 1.5, 2.5])
        'Kp_att_diag': [25.0, 25.0, 10.0], # Try reducing these (e.g., [15.0, 15.0, 5.0])
        'Kd_att_diag': [5.0, 5.0, 2.0]   # Try reducing these (e.g., [2.0, 2.0, 1.0])
    },
    'trajectory_params': {
        'hover_pos': [0.0, 0.0, 5.0],
        'circle_radius': 3.0,       # Try smaller radius?
        'circle_freq_hz': 0.1,      # Try lower frequency?
        'circle_height': 5.0,
        'bob_amp': 2.0,
        'bob_freq_hz': 0.2,
        'bob_center_z': 5.0,
        'spiral_radius': 3.0,       # Try smaller radius?
        'spiral_freq_hz': 0.1,      # Try lower frequency?
        'spiral_vert_amp': 2.0,
        'spiral_vert_freq_hz': 0.05, # Try lower frequency?
        'spiral_center_z': 5.0
    },
    'selected_trajectory': 'Hover'
}

# ---- Drone Physics and State Class (Add Sanity Checks) ----
class Drone:
    def __init__(self,
                 initial_pos=np.array([0.0, 0.0, 1.0]),
                 initial_vel=np.array([0.0, 0.0, 0.0]),
                 initial_att=np.array([0.0, 0.0, 0.0]),
                 initial_omega=np.array([0.0, 0.0, 0.0]),
                 mass=1.0,
                 inertia_diag=np.array([0.01, 0.01, 0.02]),
                 g=9.81,
                 max_thrust_per_prop=2.0,
                 num_propellers=4
                 ):
        # --- Initial State ---
        self.initial_pos = np.array(initial_pos)
        self.initial_vel = np.array(initial_vel)
        self.initial_att = np.array(initial_att)
        self.initial_omega = np.array(initial_omega)

        # --- Parameters ---
        self.mass = mass
        self.I = np.diag(inertia_diag)
        self._update_inertia_inv()
        self.g = g
        self.num_propellers = num_propellers
        self.max_thrust_per_prop = max_thrust_per_prop
        self.max_total_thrust = self.num_propellers * self.max_thrust_per_prop

        # --- Current State ---
        self.reset()

    def _update_rotation_matrix(self):
        try:
            # Ensure Euler angles are finite before calculating rotation
            if not np.all(np.isfinite(self.att_euler)):
                print("Warning: Non-finite Euler angles detected. Resetting attitude.")
                self.att_euler = self.initial_att.copy() # Or np.zeros(3)
                self.omega = self.initial_omega.copy()   # Reset omega too

            self.R = Rotation.from_euler('ZYX', self.att_euler[::-1]).as_matrix()

            # Check if Rotation Matrix itself is valid
            if not np.all(np.isfinite(self.R)):
                 print("Warning: Resulting rotation matrix R is invalid. Resetting attitude.")
                 self.att_euler = self.initial_att.copy()
                 self.omega = self.initial_omega.copy()
                 # Recalculate R with reset values
                 self.R = Rotation.from_euler('ZYX', self.att_euler[::-1]).as_matrix()

        except ValueError as e:
             print(f"Error calculating rotation matrix: {e}. Resetting attitude.")
             # Handle potential errors from Rotation if angles are extreme but finite
             self.att_euler = self.initial_att.copy()
             self.omega = self.initial_omega.copy()
             self.R = Rotation.from_euler('ZYX', self.att_euler[::-1]).as_matrix()


    def _omega_to_euler_rate(self):
        roll, pitch, yaw = self.att_euler
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        tp = np.tan(pitch)

        # Check for gimbal lock condition and handle potential division by zero/large numbers
        if abs(cp) < 1e-7:
            # At gimbal lock, avoid large numbers. Euler rates are ill-defined.
            # One option is to return zeros or a small value, but this isn't physically perfect.
            # Using Quaternions avoids this entire problem.
            print("Warning: Near gimbal lock (pitch ~ +/- 90 deg). Euler rates may be inaccurate.")
            # Clamp tp to prevent extreme values if cp is tiny but not zero
            tp = np.sign(tp) * 1e7 if abs(tp) > 1e7 else tp
            cp = np.sign(cp) * 1e-7 # Use the small value

        # Check if omega is finite
        if not np.all(np.isfinite(self.omega)):
             print("Warning: Non-finite omega detected in _omega_to_euler_rate. Returning zero rates.")
             return np.zeros(3)

        W = np.array([[1, sr * tp, cr * tp],
                      [0, cr,      -sr],
                      [0, sr / cp, cr / cp]])

        euler_rates = W @ self.omega

        # Check if calculated rates are finite
        if not np.all(np.isfinite(euler_rates)):
            print("Warning: Non-finite Euler rates calculated. Clamping.")
            # Clamp to large values instead of NaN/inf
            euler_rates = np.clip(euler_rates, -100, 100) # Adjust clamp range as needed

        return euler_rates


    def _update_inertia_inv(self):
         if np.any(np.diag(self.I) <= 1e-9): # Use small threshold instead of 0
             print("Warning: Inertia diagonal elements must be positive. Using previous/default.")
             if not hasattr(self, 'I_inv') or np.any(np.diag(self.I) <= 1e-9):
                 self.I = np.diag([0.01, 0.01, 0.02])
         try:
             self.I_inv = np.linalg.inv(self.I)
             if not np.all(np.isfinite(self.I_inv)):
                  raise np.linalg.LinAlgError("Inverse Inertia contains non-finite values.")
         except np.linalg.LinAlgError as e:
             print(f"Warning: Inertia matrix inversion failed ({e}). Using pseudo-inverse.")
             self.I_inv = np.linalg.pinv(self.I)
             if not np.all(np.isfinite(self.I_inv)):
                  print("ERROR: Pseudo-inverse of Inertia also invalid. Check Inertia values.")
                  # Fallback to a safe default if pseudo-inverse also fails
                  self.I = np.diag([0.01, 0.01, 0.02])
                  self.I_inv = np.linalg.inv(self.I)


    def reset(self):
        """Resets the drone state to initial conditions."""
        self.pos = self.initial_pos.copy()
        self.vel = self.initial_vel.copy()
        self.att_euler = self.initial_att.copy()
        self.omega = self.initial_omega.copy()
        # Ensure R is updated correctly on reset
        self._update_rotation_matrix()
        # Sanity check after reset
        if not np.all(np.isfinite(self.R)):
             print("ERROR: R matrix invalid immediately after reset!")
             # Handle this critical error, maybe exit or use identity matrix
             self.R = np.identity(3)

    def set_params(self, g=None, inertia_diag=None, max_thrust=None, num_prop=None):
        """Allows updating parameters dynamically."""
        params_changed = False
        if g is not None and self.g != g:
            self.g = g
            params_changed = True
        if inertia_diag is not None:
            new_I_diag = np.array(inertia_diag)
            if new_I_diag.shape == (3,) and np.any(self.I.diagonal() != new_I_diag):
                if np.all(new_I_diag > 1e-9): # Check > 0
                    self.I = np.diag(new_I_diag)
                    self._update_inertia_inv() # Recalculate inverse
                    params_changed = True
                else:
                     print("Warning: Inertia diagonal elements must be positive.")
            elif new_I_diag.shape != (3,):
                 print("Warning: Inertia requires 3 diagonal elements [Ixx, Iyy, Izz]. Not updated.")

        if max_thrust is not None and self.max_total_thrust != max_thrust:
             self.max_total_thrust = max(0, max_thrust)
             if self.num_propellers > 0:
                 self.max_thrust_per_prop = self.max_total_thrust / self.num_propellers
             else: self.max_thrust_per_prop = 0
             params_changed = True

        if num_prop is not None and self.num_propellers != num_prop:
             self.num_propellers = max(1, int(num_prop))
             if self.num_propellers > 0:
                 self.max_thrust_per_prop = self.max_total_thrust / self.num_propellers
             else: self.max_thrust_per_prop = 0
             params_changed = True

        return params_changed

    def update(self, dt, total_thrust, torques, perturbations=None):
        """Updates the drone state based on inputs and physics."""

        # --- Input Sanity Checks ---
        if not np.isfinite(total_thrust):
            print("Warning: Received non-finite total_thrust command. Setting to 0.")
            total_thrust = 0.0
        if not np.all(np.isfinite(torques)):
            print("Warning: Received non-finite torques command. Setting to [0,0,0].")
            torques = np.zeros(3)

        # Clamp Inputs
        total_thrust = np.clip(total_thrust, 0, self.max_total_thrust)
        # Torques should be clamped by the controller, but we can double-check here if needed

        # --- State Sanity Check (Start of Update) ---
        if not np.all(np.isfinite(self.pos)) or \
           not np.all(np.isfinite(self.vel)) or \
           not np.all(np.isfinite(self.att_euler)) or \
           not np.all(np.isfinite(self.omega)):
            print("ERROR: Drone state is non-finite at start of update step! Resetting state.")
            self.reset()
            return # Skip the rest of the update for this step

        # --- Forces ---
        F_gravity = np.array([0, 0, -self.mass * self.g])
        F_thrust_body = np.array([0, 0, total_thrust])

        # Ensure R is valid before use
        if not hasattr(self, 'R') or not np.all(np.isfinite(self.R)):
             print("Warning: Invalid R matrix in update step. Using identity.")
             self.R = np.identity(3) # Use identity as a fallback
             # Consider resetting attitude as well
             # self.att_euler = self.initial_att.copy()
             # self.omega = self.initial_omega.copy()


        F_thrust_inertial = self.R @ F_thrust_body

        F_perturb_inertial = np.zeros(3)
        if perturbations and 'force' in perturbations:
            F_perturb_inertial = np.array(perturbations['force'])
            if not np.all(np.isfinite(F_perturb_inertial)): F_perturb_inertial = np.zeros(3)

        F_total = F_gravity + F_thrust_inertial + F_perturb_inertial

        # --- Linear Dynamics ---
        linear_accel = F_total / self.mass

        # Check calculated acceleration
        if not np.all(np.isfinite(linear_accel)):
             print("Warning: Non-finite linear acceleration calculated. Setting to zero.")
             linear_accel = np.zeros(3)

        self.vel += linear_accel * dt
        self.pos += self.vel * dt

        # Check resulting velocity and position
        if not np.all(np.isfinite(self.vel)):
             print("Warning: Non-finite velocity detected. Clamping/Resetting velocity.")
             # Option 1: Clamp
             # self.vel = np.clip(self.vel, -100, 100) # Adjust limits
             # Option 2: Reset
             self.vel = self.initial_vel.copy()
        if not np.all(np.isfinite(self.pos)):
             print("Warning: Non-finite position detected. Resetting position.")
             self.pos = self.initial_pos.copy()


        # Ground constraint
        if self.pos[2] < 0:
            self.pos[2] = 0
            self.vel[2] = max(0, self.vel[2])

        # --- Torques ---
        tau_ctrl = np.array(torques) # Already checked/clamped?

        tau_perturb_body = np.zeros(3)
        if perturbations and 'torque' in perturbations:
            tau_perturb_body = np.array(perturbations['torque'])
            if not np.all(np.isfinite(tau_perturb_body)): tau_perturb_body = np.zeros(3)

        # Gyroscopic effect - Check omega and I first
        if not np.all(np.isfinite(self.omega)) or \
           not np.all(np.isfinite(self.I)) or \
           not np.all(np.isfinite(self.I @ self.omega)):
             print("Warning: Non-finite values for gyroscopic calculation. Setting tau_gyro to zero.")
             tau_gyro = np.zeros(3)
        else:
             tau_gyro = np.cross(self.omega, self.I @ self.omega)
             if not np.all(np.isfinite(tau_gyro)):
                  print("Warning: Non-finite gyroscopic torque calculated. Setting to zero.")
                  tau_gyro = np.zeros(3)


        tau_total = tau_ctrl - tau_gyro + tau_perturb_body

        # --- Rotational Dynamics ---
        # Check components before calculating angular_accel
        if not np.all(np.isfinite(tau_total)) or \
           not hasattr(self, 'I_inv') or \
           not np.all(np.isfinite(self.I_inv)):
            print("Warning: Non-finite inputs to angular acceleration calculation. Setting accel to zero.")
            angular_accel = np.zeros(3)
        else:
            angular_accel = self.I_inv @ tau_total
            if not np.all(np.isfinite(angular_accel)):
                 print("Warning: Non-finite angular acceleration calculated. Setting to zero.")
                 angular_accel = np.zeros(3)

        self.omega += angular_accel * dt

        # Check omega after update
        if not np.all(np.isfinite(self.omega)):
             print("Warning: Non-finite omega after update. Clamping/Resetting omega.")
             # self.omega = np.clip(self.omega, -100, 100) # Clamp (adjust limits)
             self.omega = self.initial_omega.copy() # Reset


        # --- Update Orientation ---
        euler_rates = self._omega_to_euler_rate() # Already has checks
        self.att_euler += euler_rates * dt

        # Check Euler angles after update
        if not np.all(np.isfinite(self.att_euler)):
             print("Warning: Non-finite Euler angles after update. Resetting attitude.")
             self.att_euler = self.initial_att.copy()
             self.omega = self.initial_omega.copy() # Reset omega too

        # Wrap Yaw angle (optional but good practice)
        self.att_euler[2] = (self.att_euler[2] + np.pi) % (2 * np.pi) - np.pi

        # Update the rotation matrix for the next step (already has checks)
        self._update_rotation_matrix()


# ---- Controller with Trajectory Following (Add Torque Clamping) ----
class TrajectoryPDController:
    def __init__(self, drone_mass, drone_g, gains, traj_params, max_torque_cfg):
        # Control Gains
        self.Kp_pos = np.diag(gains['Kp_pos_diag'])
        self.Kd_pos = np.diag(gains['Kd_pos_diag'])
        self.Kp_att = np.diag(gains['Kp_att_diag'])
        self.Kd_att = np.diag(gains['Kd_att_diag'])

        # Max Torques for Clamping
        self.max_torque = np.array(max_torque_cfg)

        # Trajectory Parameters
        self.traj_params = traj_params

        # Current Target State
        self.target_pos = np.array(self.traj_params['hover_pos'])
        self.target_vel = np.zeros(3)
        self.target_att_euler = np.array([0.0, 0.0, 0.0])

        # Drone Parameters
        self.mass = drone_mass
        self.g = drone_g

    def update_drone_params(self, mass, g):
        self.mass = mass
        self.g = g

    def update_gains(self, new_gains):
        self.Kp_pos = np.diag(new_gains['Kp_pos_diag'])
        self.Kd_pos = np.diag(new_gains['Kd_pos_diag'])
        self.Kp_att = np.diag(new_gains['Kp_att_diag'])
        self.Kd_att = np.diag(new_gains['Kd_att_diag'])

    def update_trajectory_params(self, new_traj_params):
        self.traj_params = new_traj_params

    def update_max_torque(self, max_torque_cfg):
         self.max_torque = np.array(max_torque_cfg)

    def update_target(self, time, mode):
        params = self.traj_params
        omega_circ = 2 * np.pi * params['circle_freq_hz']
        omega_bob = 2 * np.pi * params['bob_freq_hz']
        omega_spiral = 2 * np.pi * params['spiral_freq_hz']
        omega_spiral_vert = 2 * np.pi * params['spiral_vert_freq_hz']

        if mode == 'Hover':
            self.target_pos = np.array(params['hover_pos'])
            self.target_vel = np.zeros(3)
        elif mode == 'Circle':
            radius = params['circle_radius']
            x = radius * np.cos(omega_circ * time)
            y = radius * np.sin(omega_circ * time)
            z = params['circle_height']
            vx = -radius * omega_circ * np.sin(omega_circ * time)
            vy = radius * omega_circ * np.cos(omega_circ * time)
            vz = 0.0
            self.target_pos = np.array([x, y, z])
            self.target_vel = np.array([vx, vy, vz])
        elif mode == 'Bobbing':
            amp = params['bob_amp']
            z = params['bob_center_z'] + amp * np.sin(omega_bob * time)
            vz = amp * omega_bob * np.cos(omega_bob * time)
            self.target_pos = np.array([0.0, 0.0, z])
            self.target_vel = np.array([0.0, 0.0, vz])
        elif mode == 'Spiral':
            radius = params['spiral_radius']
            vert_amp = params['spiral_vert_amp']
            x = radius * np.cos(omega_spiral * time)
            y = radius * np.sin(omega_spiral * time)
            z = params['spiral_center_z'] + vert_amp * np.sin(omega_spiral_vert * time)
            vx = -radius * omega_spiral * np.sin(omega_spiral * time)
            vy = radius * omega_spiral * np.cos(omega_spiral * time)
            vz = vert_amp * omega_spiral_vert * np.cos(omega_spiral_vert * time)
            self.target_pos = np.array([x, y, z])
            self.target_vel = np.array([vx, vy, vz])
        else:
            self.target_pos = np.array(params['hover_pos'])
            self.target_vel = np.zeros(3)

        # Check for NaNs in target (e.g., if time becomes huge)
        if not np.all(np.isfinite(self.target_pos)) or not np.all(np.isfinite(self.target_vel)):
             print("Warning: Non-finite target generated. Resetting target to hover.")
             self.target_pos = np.array(params['hover_pos'])
             self.target_vel = np.zeros(3)

        self.target_att_euler = np.array([0.0, 0.0, 0.0])


    def calculate_control(self, current_pos, current_vel, current_att_euler, current_omega, R_matrix):

        # --- Sanity Check Inputs ---
        if not np.all(np.isfinite(current_pos)) or \
           not np.all(np.isfinite(current_vel)) or \
           not np.all(np.isfinite(current_att_euler)) or \
           not np.all(np.isfinite(current_omega)) or \
           not np.all(np.isfinite(R_matrix)):
             print("ERROR: Controller received non-finite state or R_matrix! Returning zero commands.")
             return 0.0, np.zeros(3)


        # --- Position Control ---
        pos_error = self.target_pos - current_pos
        vel_error = self.target_vel - current_vel

        # Check errors
        if not np.all(np.isfinite(pos_error)) or not np.all(np.isfinite(vel_error)):
             print("Warning: Non-finite errors in controller. Using zero errors.")
             pos_error = np.zeros(3)
             vel_error = np.zeros(3)

        # Desired force
        F_desired = (self.Kp_pos @ pos_error +
                     self.Kd_pos @ vel_error +
                     np.array([0, 0, self.mass * self.g]))

        if not np.all(np.isfinite(F_desired)):
             print("Warning: Non-finite F_desired calculated. Using gravity compensation only.")
             F_desired = np.array([0, 0, self.mass * self.g])


        # Required total thrust
        body_z_axis = R_matrix[:, 2]
        total_thrust = F_desired @ body_z_axis

        # --- Attitude Control ---
        F_norm = np.linalg.norm(F_desired)
        if F_norm > 1e-6:
            z_b_des = F_desired / F_norm
        else:
            z_b_des = np.array([0, 0, 1])

        # Simplified yaw control (world X as reference)
        x_c_des = np.array([1, 0, 0])
        y_b_des_cross = np.cross(z_b_des, x_c_des)
        y_b_norm = np.linalg.norm(y_b_des_cross)

        if y_b_norm < 1e-6:
             # z_b_des is aligned with x_c_des (world X), use world Y for cross product
             y_b_des_cross = np.cross(z_b_des, np.array([0,1,0]))
             y_b_norm = np.linalg.norm(y_b_des_cross)
             # If still zero norm (shouldn't happen if z_b_des is unit vec), fallback
             if y_b_norm < 1e-6:
                  print("Warning: Could not determine y_b_des. Using default orientation.")
                  R_des = np.identity(3) # Fallback if axes are degenerate
             else:
                  y_b_des = y_b_des_cross / y_b_norm
                  x_b_des = np.cross(y_b_des, z_b_des)
                  R_des = np.vstack((x_b_des, y_b_des, z_b_des)).T
        else:
             y_b_des = y_b_des_cross / y_b_norm
             x_b_des = np.cross(y_b_des, z_b_des)
             R_des = np.vstack((x_b_des, y_b_des, z_b_des)).T

        # Calculate attitude error matrix
        R_err = R_des @ R_matrix.T

        # --- Check R_err before converting ---
        if not np.all(np.isfinite(R_err)):
            print("Warning: R_err matrix contains non-finite values. Skipping attitude control step.")
            att_error_vec = np.zeros(3) # Use zero error
        else:
            # Convert rotation matrix error to axis-angle representation
            try:
                # Use scipy's robust conversion
                rot_err = Rotation.from_matrix(R_err)
                att_error_vec = rot_err.as_rotvec()
                if not np.all(np.isfinite(att_error_vec)):
                     print("Warning: Non-finite att_error_vec calculated. Using zero error.")
                     att_error_vec = np.zeros(3)
            except ValueError as e:
                 # Handles cases where R_err is slightly non-orthogonal due to float errors
                 # or potentially more serious issues if it's very far from SO(3)
                 print(f"Warning: ValueError converting R_err to rotation vector: {e}. Using zero error.")
                 # Attempt to find nearest SO(3) matrix using SVD (may still fail if NaN/inf)
                 try:
                     U, _, Vh = np.linalg.svd(R_err)
                     R_err_approx = U @ Vh
                     # Ensure determinant is +1
                     if np.linalg.det(R_err_approx) < 0:
                         Vh[-1, :] *= -1 # Flip the sign of the last row of Vh
                         R_err_approx = U @ Vh

                     if np.all(np.isfinite(R_err_approx)):
                          rot_err = Rotation.from_matrix(R_err_approx)
                          att_error_vec = rot_err.as_rotvec()
                          if not np.all(np.isfinite(att_error_vec)): att_error_vec = np.zeros(3) # Check again
                     else:
                          att_error_vec = np.zeros(3) # Fallback if approximation failed
                 except np.linalg.LinAlgError:
                      print("SVD failed during R_err approximation. Using zero error.")
                      att_error_vec = np.zeros(3) # Fallback
                 except Exception as ex:
                      print(f"Unexpected error during R_err approximation: {ex}. Using zero error.")
                      att_error_vec = np.zeros(3)


        # Angular Velocity Error
        omega_error = -current_omega # Target omega is zero
        if not np.all(np.isfinite(omega_error)):
             print("Warning: Non-finite omega_error. Using zero error.")
             omega_error = np.zeros(3)

        # Desired Torques (PD control)
        torques_calculated = self.Kp_att @ att_error_vec + self.Kd_att @ omega_error

        if not np.all(np.isfinite(torques_calculated)):
             print("Warning: Non-finite torques calculated before clamping. Setting to zero.")
             torques_calculated = np.zeros(3)

        # --- Clamp Torques ---
        torques = np.clip(torques_calculated, -self.max_torque, self.max_torque)

        # Clamp Thrust as well (redundant with drone update, but safe)
        total_thrust = np.clip(total_thrust, 0, None) # Ensure non-negative, max handled by drone

        return total_thrust, torques


# ---- Config File Functions (Unchanged) ----
def load_config(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            # Merge with defaults, ensuring all keys exist
            merged_config = DEFAULT_CONFIG.copy()

            def recursive_update(default_dict, loaded_dict):
                for key, value in loaded_dict.items():
                    if key in default_dict:
                        if isinstance(default_dict[key], dict) and isinstance(value, dict):
                            recursive_update(default_dict[key], value)
                        # Basic type check or allow overwrite
                        elif isinstance(value, type(default_dict[key])) or default_dict[key] is None:
                             default_dict[key] = value
                        # Handle list case (e.g., initial state, gains) - ensure list type
                        elif isinstance(default_dict[key], list) and isinstance(value, list):
                             # Optional: Check length if needed default_dict[key] = value
                             default_dict[key] = value

                        else:
                            print(f"Warning: Type mismatch or incompatible types for key '{key}'. Loaded: {type(value)}, Default: {type(default_dict[key])}. Using default.")
                    else:
                        print(f"Warning: Unknown key '{key}' in config file section. Ignored.")

            recursive_update(merged_config, config)

            # Ensure essential nested dictionaries exist if file was minimal
            for key in ['controller_gains', 'trajectory_params']:
                 if key not in merged_config: merged_config[key] = DEFAULT_CONFIG[key]

            # Ensure max_torque exists
            if 'max_torque' not in merged_config:
                 merged_config['max_torque'] = DEFAULT_CONFIG['max_torque']


            print(f"Loaded configuration from {filepath}")
            return merged_config
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filepath}. Using default config.")
            return DEFAULT_CONFIG.copy()
        except Exception as e:
            print(f"Error loading config: {e}. Using default config.")
            return DEFAULT_CONFIG.copy()
    else:
        print(f"Config file {filepath} not found. Using default config.")
        return DEFAULT_CONFIG.copy()

def save_config(filepath, config_data):
    def convert_numpy_to_list(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, dict): return {k: convert_numpy_to_list(v) for k, v in item.items()}
        if isinstance(item, list): return [convert_numpy_to_list(i) for i in item]
        return item
    config_to_save = convert_numpy_to_list(config_data)
    try:
        with open(filepath, 'w') as f: json.dump(config_to_save, f, indent=4)
        print(f"Saved configuration to {filepath}")
    except Exception as e: print(f"Error saving config: {e}")


# ---- Simulation Setup (Pass max_torque to controller) ----
DT = 0.02
ANIM_INTERVAL = 20

current_config = load_config(CONFIG_FILE)

drone_params = { k: current_config[k] for k in ['initial_pos', 'initial_vel', 'initial_att', 'initial_omega', 'mass', 'inertia_diag', 'g', 'max_thrust_per_prop', 'num_propellers'] }
drone = Drone(**drone_params)

controller = TrajectoryPDController(drone.mass, drone.g,
                                  current_config['controller_gains'],
                                  current_config['trajectory_params'],
                                  current_config['max_torque']) # Pass max_torque

sim_time = 0.0
history_pos = [drone.pos.copy()]
history_target = [controller.target_pos.copy()]
history_time = [sim_time]
max_history = 300 # Increased history slightly
current_trajectory_mode = current_config.get('selected_trajectory', 'Hover')

# ---- Plotting Setup (Unchanged) ----
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2)
ax3d = fig.add_subplot(gs[0, 0], projection='3d')
ax3d.set_title("Isometric View"), ax3d.set_xlabel("X (m)"), ax3d.set_ylabel("Y (m)"), ax3d.set_zlabel("Z (m)")
line3d, = ax3d.plot([], [], [], 'b-', label="Trajectory")
point3d, = ax3d.plot([], [], [], 'ro', markersize=5, label="Drone")
target3d, = ax3d.plot([], [], [], 'gx', markersize=7, alpha=0.7, label="Target") # Added alpha
ax3d.set_xlim(-6, 6), ax3d.set_ylim(-6, 6), ax3d.set_zlim(0, 10)
ax3d.legend(fontsize='small')
ax3d.grid(True)
ax_xy = fig.add_subplot(gs[0, 1])
ax_xy.set_title("XY View (Top Down)"), ax_xy.set_xlabel("X (m)"), ax_xy.set_ylabel("Y (m)")
line_xy, = ax_xy.plot([], [], 'b-')
point_xy, = ax_xy.plot([], [], 'ro', markersize=5)
target_xy, = ax_xy.plot([], [], 'gx', markersize=7, alpha=0.7)
ax_xy.set_xlim(-6, 6), ax_xy.set_ylim(-6, 6), ax_xy.set_aspect('equal', adjustable='box'), ax_xy.grid(True)
ax_yz = fig.add_subplot(gs[1, 0])
ax_yz.set_title("YZ View (Side)"), ax_yz.set_xlabel("Y (m)"), ax_yz.set_ylabel("Z (m)")
line_yz, = ax_yz.plot([], [], 'b-')
point_yz, = ax_yz.plot([], [], 'ro', markersize=5)
target_yz, = ax_yz.plot([], [], 'gx', markersize=7, alpha=0.7)
ax_yz.set_xlim(-6, 6), ax_yz.set_ylim(0, 10), ax_yz.grid(True)
ax_xz = fig.add_subplot(gs[1, 1])
ax_xz.set_title("XZ View (Front)"), ax_xz.set_xlabel("X (m)"), ax_xz.set_ylabel("Z (m)")
line_xz, = ax_xz.plot([], [], 'b-')
point_xz, = ax_xz.plot([], [], 'ro', markersize=5)
target_xz, = ax_xz.plot([], [], 'gx', markersize=7, alpha=0.7)
ax_xz.set_xlim(-6, 6), ax_xz.set_ylim(0, 10), ax_xz.grid(True)
plt.tight_layout(rect=[0, 0.15, 1, 0.95])

# ---- GUI Widgets (Add Max Torque Display/Input? - For now, just use config) ----
axcolor = 'lightgoldenrodyellow'
widget_x_start = 0.1
widget_y_start = 0.12
widget_height = 0.018
widget_spacing = 0.005
slider_width = 0.55
button_width = 0.08
radio_width = 0.15

# Sliders
ax_grav = plt.axes([widget_x_start, widget_y_start - 1*(widget_height+widget_spacing), slider_width, widget_height], facecolor=axcolor)
slider_grav = Slider(ax_grav, 'Gravity', 0.0, 20.0, valinit=drone.g, valstep=0.1)
ax_thrust = plt.axes([widget_x_start, widget_y_start - 2*(widget_height+widget_spacing), slider_width, widget_height], facecolor=axcolor)
slider_thrust = Slider(ax_thrust, 'Max Thrust', 1.0, 40.0, valinit=drone.max_total_thrust, valstep=0.5)
ax_prop = plt.axes([widget_x_start, widget_y_start - 3*(widget_height+widget_spacing), slider_width, widget_height], facecolor=axcolor)
slider_prop = Slider(ax_prop, 'Num Props', 1, 12, valinit=drone.num_propellers, valstep=1)
# Inertia Sliders
ax_ixx = plt.axes([widget_x_start, widget_y_start - 4*(widget_height+widget_spacing), slider_width/3 - widget_spacing, widget_height], facecolor=axcolor)
slider_ixx = Slider(ax_ixx, 'Ixx', 0.001, 0.1, valinit=drone.I[0,0], valfmt='%0.3f')
ax_iyy = plt.axes([widget_x_start + slider_width/3, widget_y_start - 4*(widget_height+widget_spacing), slider_width/3 - widget_spacing, widget_height], facecolor=axcolor)
slider_iyy = Slider(ax_iyy, 'Iyy', 0.001, 0.1, valinit=drone.I[1,1], valfmt='%0.3f')
ax_izz = plt.axes([widget_x_start + 2*slider_width/3, widget_y_start - 4*(widget_height+widget_spacing), slider_width/3 - widget_spacing, widget_height], facecolor=axcolor)
slider_izz = Slider(ax_izz, 'Izz', 0.001, 0.1, valinit=drone.I[2,2], valfmt='%0.3f')

# Buttons
button_x_start = widget_x_start + slider_width + 0.05
ax_reset = plt.axes([button_x_start, widget_y_start - 1*(widget_height+widget_spacing), button_width, widget_height])
button_reset = Button(ax_reset, 'Reset', color=axcolor, hovercolor='0.975')
ax_load = plt.axes([button_x_start, widget_y_start - 2*(widget_height+widget_spacing), button_width, widget_height])
button_load = Button(ax_load, 'Load Cfg', color=axcolor, hovercolor='0.975')
ax_save = plt.axes([button_x_start, widget_y_start - 3*(widget_height+widget_spacing), button_width, widget_height])
button_save = Button(ax_save, 'Save Cfg', color=axcolor, hovercolor='0.975')

# Radio Buttons
ax_radio = plt.axes([button_x_start + button_width + 0.02, widget_y_start - 4.5*(widget_height+widget_spacing), radio_width, 4.5*widget_height], facecolor=axcolor)
traj_options_list = ('Hover', 'Circle', 'Bobbing', 'Spiral')
try:
    active_traj_index = traj_options_list.index(current_trajectory_mode)
except ValueError:
    active_traj_index = 0 # Default to Hover if not found
radio_traj = RadioButtons(ax_radio, traj_options_list, active=active_traj_index)
radio_traj.activecolor = 'lightblue'

# ---- GUI Callbacks (Update controller max_torque on load) ----
def update_gui_sliders(config):
    slider_grav.set_val(config['g'])
    inertia_diag = config.get('inertia_diag', np.diag(config.get('inertia_matrix', drone.I)).tolist())
    slider_ixx.set_val(inertia_diag[0])
    slider_iyy.set_val(inertia_diag[1])
    slider_izz.set_val(inertia_diag[2])
    max_total_t = config['num_propellers'] * config['max_thrust_per_prop']
    slider_thrust.set_val(max_total_t)
    slider_prop.set_val(config['num_propellers'])
    traj_options = [label.get_text() for label in radio_traj.labels]
    try:
        active_index = traj_options.index(config.get('selected_trajectory', 'Hover'))
        radio_traj.set_active(active_index)
    except ValueError: radio_traj.set_active(0)

def update_params(val):
    inertia_diag = [slider_ixx.val, slider_iyy.val, slider_izz.val]
    drone_updated = drone.set_params(g=slider_grav.val,
                                     inertia_diag=inertia_diag,
                                     max_thrust=slider_thrust.val,
                                     num_prop=slider_prop.val)
    if drone_updated: controller.update_drone_params(drone.mass, drone.g)

slider_grav.on_changed(update_params)
slider_thrust.on_changed(update_params)
slider_prop.on_changed(update_params)
slider_ixx.on_changed(update_params)
slider_iyy.on_changed(update_params)
slider_izz.on_changed(update_params)

def reset_sim(event):
    global sim_time, history_pos, history_target, history_time
    # Ensure drone state is valid before trying to use it
    try:
         drone.reset()
    except Exception as e:
         print(f"Error during drone reset: {e}. Attempting recovery.")
         # Force re-initialization if reset fails badly
         drone.__init__(**drone_params)

    controller.update_target(0.0, current_trajectory_mode)
    controller.update_drone_params(drone.mass, drone.g)
    sim_time = 0.0
    history_pos = [drone.pos.copy()]
    history_target = [controller.target_pos.copy()]
    history_time = [sim_time]
    line3d.set_data_3d([], [], []), point3d.set_data_3d([], [], []), target3d.set_data_3d([], [], [])
    line_xy.set_data([], []), point_xy.set_data([], []), target_xy.set_data([], [])
    line_yz.set_data([], []), point_yz.set_data([], []), target_yz.set_data([], [])
    line_xz.set_data([], []), point_xz.set_data([], []), target_xz.set_data([], [])
    fig.canvas.draw_idle()

button_reset.on_clicked(reset_sim)

def load_sim_config(event):
    global current_config, current_trajectory_mode
    loaded_config = load_config(CONFIG_FILE)
    current_config = loaded_config

    # Update Drone parameters
    drone_params_loaded = { k: current_config[k] for k in ['initial_pos', 'initial_vel', 'initial_att', 'initial_omega', 'mass', 'inertia_diag', 'g', 'max_thrust_per_prop', 'num_propellers'] }
    try:
         drone.__init__(**drone_params_loaded)
    except Exception as e:
         print(f"Error re-initializing drone from config: {e}. Check config file.")
         # Optionally fall back to defaults or keep existing drone object
         return

    # Update Controller parameters (including max_torque)
    controller.update_drone_params(drone.mass, drone.g)
    controller.update_gains(current_config['controller_gains'])
    controller.update_trajectory_params(current_config['trajectory_params'])
    controller.update_max_torque(current_config['max_torque']) # Update torque limits

    update_gui_sliders(current_config)
    current_trajectory_mode = current_config.get('selected_trajectory', 'Hover')
    traj_options = [label.get_text() for label in radio_traj.labels]
    try:
        active_index = traj_options.index(current_trajectory_mode)
        radio_traj.set_active(active_index)
    except ValueError: radio_traj.set_active(0), (current_trajectory_mode := traj_options[0])
    reset_sim(None)

button_load.on_clicked(load_sim_config)

def save_sim_config(event):
    # Get selected trajectory label text robustly
    selected_label_text = 'Hover' # Default
    for label in radio_traj.labels:
        if radio_traj.value_selected == label.get_text(): # Check if this label is the selected one
            selected_label_text = label.get_text()
            break

    config_to_save = {
        'initial_pos': drone.initial_pos.tolist(),
        'initial_vel': drone.initial_vel.tolist(),
        'initial_att': drone.initial_att.tolist(),
        'initial_omega': drone.initial_omega.tolist(),
        'mass': drone.mass,
        'inertia_diag': drone.I.diagonal().tolist(),
        'g': drone.g,
        'max_thrust_per_prop': drone.max_thrust_per_prop,
        'num_propellers': drone.num_propellers,
        'max_torque': controller.max_torque.tolist(), # Save max torque limits
        'controller_gains': {
             'Kp_pos_diag': np.diag(controller.Kp_pos).tolist(),
             'Kd_pos_diag': np.diag(controller.Kd_pos).tolist(),
             'Kp_att_diag': np.diag(controller.Kp_att).tolist(),
             'Kd_att_diag': np.diag(controller.Kd_att).tolist()
        },
        'trajectory_params': controller.traj_params,
        'selected_trajectory': selected_label_text
    }
    save_config(CONFIG_FILE, config_to_save)

button_save.on_clicked(save_sim_config)

def on_trajectory_select(label):
    global current_trajectory_mode
    current_trajectory_mode = label
    controller.update_target(sim_time, current_trajectory_mode)
    current_config['selected_trajectory'] = label

radio_traj.on_clicked(on_trajectory_select)


# ---- Animation Function (Error handling inside) ----
animation_running = True # Flag to stop animation loop on critical error

def animate(frame):
    global sim_time, history_pos, history_target, history_time, animation_running

    if not animation_running: return [] # Stop updates if flag is false

    try:
        # --- Update Target ---
        controller.update_target(sim_time, current_trajectory_mode)

        # --- Control Input ---
        total_thrust, torques = controller.calculate_control(
            drone.pos, drone.vel, drone.att_euler, drone.omega, drone.R
        )

        # --- Perturbation Input ---
        perturb = None

        # --- Update Drone State ---
        drone.update(DT, total_thrust, torques, perturbations=perturb)
        sim_time += DT

        # --- Store History ---
        # Only store if state is valid
        if np.all(np.isfinite(drone.pos)) and np.all(np.isfinite(controller.target_pos)):
            history_pos.append(drone.pos.copy())
            history_target.append(controller.target_pos.copy())
            history_time.append(sim_time)
            if len(history_pos) > max_history:
                history_pos.pop(0)
                history_target.pop(0)
                history_time.pop(0)
        else:
             print("Skipping history storage due to non-finite state.")


        # --- Update Plots ---
        hist_np = np.array(history_pos)
        target_pos_current = controller.target_pos

        # Check if history is valid before plotting
        if hist_np.size == 0 or not np.all(np.isfinite(hist_np)):
             print("History is empty or invalid, skipping plot update.")
             # Optionally clear plots or just return existing artists
             return (line3d, point3d, target3d, line_xy, point_xy, target_xy,
                     line_yz, point_yz, target_yz, line_xz, point_xz, target_xz)


        line3d.set_data(hist_np[:, 0], hist_np[:, 1])
        line3d.set_3d_properties(hist_np[:, 2])
        point3d.set_data([drone.pos[0]], [drone.pos[1]])
        point3d.set_3d_properties([drone.pos[2]])
        if np.all(np.isfinite(target_pos_current)):
            target3d.set_data([target_pos_current[0]], [target_pos_current[1]])
            target3d.set_3d_properties([target_pos_current[2]])

        line_xy.set_data(hist_np[:, 0], hist_np[:, 1])
        point_xy.set_data([drone.pos[0]], [drone.pos[1]])
        if np.all(np.isfinite(target_pos_current)): target_xy.set_data([target_pos_current[0]], [target_pos_current[1]])

        line_yz.set_data(hist_np[:, 1], hist_np[:, 2])
        point_yz.set_data([drone.pos[1]], [drone.pos[2]])
        if np.all(np.isfinite(target_pos_current)): target_yz.set_data([target_pos_current[1]], [target_pos_current[2]])

        line_xz.set_data(hist_np[:, 0], hist_np[:, 2])
        point_xz.set_data([drone.pos[0]], [drone.pos[2]])
        if np.all(np.isfinite(target_pos_current)): target_xz.set_data([target_pos_current[0]], [target_pos_current[1]]) # Z-axis typo fixed here

        return (line3d, point3d, target3d,
                line_xy, point_xy, target_xy,
                line_yz, point_yz, target_yz,
                line_xz, point_xz, target_xz)

    except Exception as e:
        print(f"\n--- CRITICAL ERROR IN ANIMATION LOOP ---")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("-----------------------------------------")
        print("Animation stopped due to error.")
        animation_running = False # Stop the animation loop
        # Optionally, re-raise the exception if you want the program to fully halt
        # raise e
        return [] # Return empty list of artists

# ---- Run Animation ----
# Suppress the specific user warning about frames=None and cache_frame_data
import warnings
warnings.filterwarnings("ignore", message=".*frames=None which we can infer the length of.*")

ani = animation.FuncAnimation(fig, animate, frames=None, interval=ANIM_INTERVAL, blit=True, repeat=False, cache_frame_data=False) # Set cache_frame_data=False

plt.suptitle("Drone Simulation with Configuration and Trajectories")
plt.show()
