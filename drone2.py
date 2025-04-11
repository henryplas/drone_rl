import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons # Added RadioButtons
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation
import json # For config file handling
import os # To check file existence

# ---- Configuration ----
CONFIG_FILE = 'drone_config.json'
DEFAULT_CONFIG = {
    'initial_pos': [0.0, 0.0, 1.0],
    'initial_vel': [0.0, 0.0, 0.0],
    'initial_att': [0.0, 0.0, 0.0], # Euler angles: roll, pitch, yaw (rad)
    'initial_omega': [0.0, 0.0, 0.0], # Angular velocity: p, q, r (rad/s) in body frame
    'mass': 1.0, # kg
    'inertia_diag': [0.005, 0.005, 0.01], # Store diagonal Ixx, Iyy, Izz kg*m^2
    'g': 9.81, # m/s^2
    'max_thrust_per_prop': 5.0, # N
    'num_propellers': 4, # Affects total max thrust
    'controller_gains': {
        'Kp_pos_diag': [2.0, 2.0, 5.0],
        'Kd_pos_diag': [2.5, 2.5, 4.0],
        'Kp_att_diag': [50.0, 50.0, 20.0],
        'Kd_att_diag': [10.0, 10.0, 5.0]
    },
    'trajectory_params': {
        'hover_pos': [0.0, 0.0, 5.0],
        'circle_radius': 3.0,
        'circle_freq_hz': 0.1,
        'circle_height': 5.0,
        'bob_amp': 2.0,
        'bob_freq_hz': 0.2,
        'bob_center_z': 5.0,
        'spiral_radius': 3.0,
        'spiral_freq_hz': 0.1,
        'spiral_vert_amp': 2.0,
        'spiral_vert_freq_hz': 0.05,
        'spiral_center_z': 5.0
    },
    'selected_trajectory': 'Hover' # Default trajectory mode
}

# ---- Drone Physics and State Class (Mostly Unchanged) ----
class Drone:
    def __init__(self,
                 initial_pos=np.array([0.0, 0.0, 1.0]),
                 initial_vel=np.array([0.0, 0.0, 0.0]),
                 initial_att=np.array([0.0, 0.0, 0.0]),
                 initial_omega=np.array([0.0, 0.0, 0.0]),
                 mass=1.0,
                 inertia_diag=np.array([0.01, 0.01, 0.02]), # Use diagonal elements
                 g=9.81,
                 max_thrust_per_prop=2.0,
                 num_propellers=4
                 ):
        # --- Initial State (from config or defaults) ---
        self.initial_pos = np.array(initial_pos)
        self.initial_vel = np.array(initial_vel)
        self.initial_att = np.array(initial_att)
        self.initial_omega = np.array(initial_omega)

        # --- Parameters ---
        self.mass = mass
        self.I = np.diag(inertia_diag) # Create diagonal matrix
        self._update_inertia_inv() # Calculate inverse
        self.g = g
        self.num_propellers = num_propellers
        self.max_thrust_per_prop = max_thrust_per_prop
        self.max_total_thrust = self.num_propellers * self.max_thrust_per_prop

        # --- Current State ---
        self.reset() # Initialize current state

    def _update_rotation_matrix(self):
        self.R = Rotation.from_euler('ZYX', self.att_euler[::-1]).as_matrix()

    def _omega_to_euler_rate(self):
        roll, pitch, yaw = self.att_euler
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        tp = np.tan(pitch)
        if abs(cp) < 1e-6: cp = np.sign(cp) * 1e-6
        W = np.array([[1, sr * tp, cr * tp],
                      [0, cr,      -sr],
                      [0, sr / cp, cr / cp]])
        return W @ self.omega

    def _update_inertia_inv(self):
         # Handle potential non-invertibility if values are zero/negative
         if np.any(np.diag(self.I) <= 0):
             print("Warning: Inertia diagonal elements must be positive. Using previous/default.")
             # Revert to a safe default or previous value if needed
             if not hasattr(self, 'I_inv') or np.any(np.diag(self.I) <= 0): # Ensure I_inv exists
                 self.I = np.diag([0.01, 0.01, 0.02]) # Safe default
         try:
             self.I_inv = np.linalg.inv(self.I)
         except np.linalg.LinAlgError:
             print("Warning: Inertia matrix became singular. Using pseudo-inverse.")
             self.I_inv = np.linalg.pinv(self.I)


    def reset(self):
        """Resets the drone state to initial conditions."""
        self.pos = self.initial_pos.copy()
        self.vel = self.initial_vel.copy()
        self.att_euler = self.initial_att.copy()
        self.omega = self.initial_omega.copy()
        self._update_rotation_matrix()

    def set_params(self, g=None, inertia_diag=None, max_thrust=None, num_prop=None):
        """Allows updating parameters dynamically."""
        params_changed = False
        if g is not None and self.g != g:
            self.g = g
            params_changed = True
        if inertia_diag is not None:
            new_I_diag = np.array(inertia_diag)
            if new_I_diag.shape == (3,) and np.any(self.I.diagonal() != new_I_diag):
                 # Only update if valid and different
                if np.all(new_I_diag > 0):
                    self.I = np.diag(new_I_diag)
                    self._update_inertia_inv()
                    params_changed = True
                else:
                     print("Warning: Inertia diagonal elements must be positive.")
            elif new_I_diag.shape != (3,):
                 print("Warning: Inertia requires 3 diagonal elements [Ixx, Iyy, Izz]. Not updated.")

        # Update max thrust AND per-prop thrust together
        if max_thrust is not None and self.max_total_thrust != max_thrust:
            self.max_total_thrust = max(0, max_thrust) # Ensure non-negative
            if self.num_propellers > 0:
                 self.max_thrust_per_prop = self.max_total_thrust / self.num_propellers
            else:
                 self.max_thrust_per_prop = 0
            params_changed = True

        if num_prop is not None and self.num_propellers != num_prop:
             self.num_propellers = max(1, int(num_prop)) # Ensure at least 1 prop
             # Recalculate total thrust based on existing per-prop thrust OR
             # Keep total thrust and recalculate per-prop (more intuitive with slider)
             if self.num_propellers > 0:
                 self.max_thrust_per_prop = self.max_total_thrust / self.num_propellers
             else:
                  self.max_thrust_per_prop = 0 # Should not happen with max(1,...)
             params_changed = True


        return params_changed # Indicate if parameters actually changed

    def update(self, dt, total_thrust, torques, perturbations=None):
        """Updates the drone state based on inputs and physics."""
        total_thrust = np.clip(total_thrust, 0, self.max_total_thrust)
        F_gravity = np.array([0, 0, -self.mass * self.g])
        F_thrust_body = np.array([0, 0, total_thrust])
        F_thrust_inertial = self.R @ F_thrust_body
        F_perturb_inertial = np.zeros(3)
        if perturbations and 'force' in perturbations:
            F_perturb_inertial = np.array(perturbations['force'])
        F_total = F_gravity + F_thrust_inertial + F_perturb_inertial
        linear_accel = F_total / self.mass
        self.vel += linear_accel * dt
        self.pos += self.vel * dt

        if self.pos[2] < 0:
            self.pos[2] = 0
            self.vel[2] = max(0, self.vel[2])

        tau_ctrl = np.array(torques)
        tau_perturb_body = np.zeros(3)
        if perturbations and 'torque' in perturbations:
            tau_perturb_body = np.array(perturbations['torque'])
        tau_gyro = np.cross(self.omega, self.I @ self.omega)
        tau_total = tau_ctrl - tau_gyro + tau_perturb_body

        angular_accel = self.I_inv @ tau_total
        self.omega += angular_accel * dt

        euler_rates = self._omega_to_euler_rate()
        self.att_euler += euler_rates * dt
        # self.att_euler[2] = (self.att_euler[2] + np.pi) % (2 * np.pi) - np.pi # Yaw wrapping
        self._update_rotation_matrix()


# ---- Controller with Trajectory Following ----
class TrajectoryPDController:
    def __init__(self, drone_mass, drone_g, gains, traj_params):
        # Control Gains
        self.Kp_pos = np.diag(gains['Kp_pos_diag'])
        self.Kd_pos = np.diag(gains['Kd_pos_diag'])
        self.Kp_att = np.diag(gains['Kp_att_diag'])
        self.Kd_att = np.diag(gains['Kd_att_diag'])

        # Trajectory Parameters (store locally)
        self.traj_params = traj_params

        # Current Target State (updated dynamically)
        self.target_pos = np.array(self.traj_params['hover_pos']) # Initial target
        self.target_vel = np.zeros(3) # Target velocity (for feedforward/deriv term)
        self.target_att_euler = np.array([0.0, 0.0, 0.0]) # Target roll=0, pitch=0, yaw=0

        # Store drone parameters needed for control
        self.mass = drone_mass
        self.g = drone_g

    def update_drone_params(self, mass, g):
        """Update controller parameters if drone parameters change"""
        self.mass = mass
        self.g = g

    def update_gains(self, new_gains):
        """Update controller gains if loaded from config"""
        self.Kp_pos = np.diag(new_gains['Kp_pos_diag'])
        self.Kd_pos = np.diag(new_gains['Kd_pos_diag'])
        self.Kp_att = np.diag(new_gains['Kp_att_diag'])
        self.Kd_att = np.diag(new_gains['Kd_att_diag'])

    def update_trajectory_params(self, new_traj_params):
        """Update trajectory parameters if loaded from config"""
        self.traj_params = new_traj_params

    def update_target(self, time, mode):
        """Calculates the target position and velocity based on time and mode."""
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
            self.target_pos = np.array([0.0, 0.0, z]) # Bob around origin X,Y
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
        else: # Default to Hover if mode is unknown
            self.target_pos = np.array(params['hover_pos'])
            self.target_vel = np.zeros(3)

        # Keep target yaw = 0 for simplicity for now
        self.target_att_euler = np.array([0.0, 0.0, 0.0])


    def calculate_control(self, current_pos, current_vel, current_att_euler, current_omega, R_matrix):
        # --- Position Control ---
        pos_error = self.target_pos - current_pos
        # Use target velocity in derivative term
        vel_error = self.target_vel - current_vel

        # Desired force in inertial frame (PD control + Gravity Compensation + Velocity Feedforward)
        # Add a small feedforward term based on target acceleration (derivative of target_vel) ?
        # For now, let's just use target velocity in Kd term.
        F_desired = (self.Kp_pos @ pos_error +
                     self.Kd_pos @ vel_error +
                     np.array([0, 0, self.mass * self.g]))
                     # Add mass * target_acceleration here for feedforward if desired

        # Required total thrust projected onto the current body z-axis
        # F_desired needs to be produced by thrust vector T = R @ [0, 0, Thrust_total]
        # So, Thrust_total = magnitude of F_desired projected onto body z = R.T @ [0, 0, 1]
        body_z_axis = R_matrix[:, 2] # Third column of rotation matrix
        total_thrust = F_desired @ body_z_axis # Dot product gives projection magnitude

        # --- Attitude Control ---
        # Calculate desired orientation (Rotation matrix) to align body z with F_desired
        # More robust method than calculating desired Euler angles directly
        if np.linalg.norm(F_desired) > 1e-6:
            z_b_des = F_desired / np.linalg.norm(F_desired)
        else: # Avoid division by zero if desired force is tiny
            z_b_des = np.array([0, 0, 1])

        # Assuming desired yaw is 0, define desired x_b direction
        # Project inertial [1,0,0] onto the plane normal to z_b_des
        # (More complex yaw control needed for arbitrary target yaw)
        x_c_des = np.array([1, 0, 0]) # Desired heading direction (along inertial X)
        y_b_des = np.cross(z_b_des, x_c_des)
        # Normalize y_b_des, handle case where z_b_des is parallel to x_c_des
        if np.linalg.norm(y_b_des) < 1e-6:
             # If z_b_des is aligned with world X (e.g., pointing straight up/down after pitch),
             # use world Y as the reference for y_b
             y_b_des = np.cross(z_b_des, np.array([0,1,0]))
        y_b_des /= np.linalg.norm(y_b_des)
        x_b_des = np.cross(y_b_des, z_b_des)
        # Desired Rotation Matrix
        R_des = np.vstack((x_b_des, y_b_des, z_b_des)).T

        # Calculate attitude error using rotation matrices
        # Error matrix: R_err = R_des @ R_current.T
        R_err = R_des @ R_matrix.T
        # Convert rotation matrix error to axis-angle representation -> angle error vector
        # (Small angle approximation: error_angles approx [R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]] / 2)
        # Using scipy Rotation object is safer:
        rot_err = Rotation.from_matrix(R_err)
        att_error_vec = rot_err.as_rotvec() # Gives axis-angle vector (angle * axis)

        # Angular Velocity Error (target omega is zero for stabilization)
        omega_error = -current_omega

        # Desired Torques (PD control on attitude error vector and omega error)
        # Note: Kp_att and Kd_att gains might need different tuning for this formulation
        torques = self.Kp_att @ att_error_vec + self.Kd_att @ omega_error

        return total_thrust, torques

# ---- Config File Functions ----
def load_config(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            # Validate / merge with defaults for missing keys
            merged_config = DEFAULT_CONFIG.copy()
            # Basic merge (won't handle nested dict updates well without more logic)
            for key, value in config.items():
                 if key in merged_config:
                      # Basic type check (can be expanded)
                     if isinstance(value, type(merged_config[key])):
                         merged_config[key] = value
                     elif isinstance(merged_config[key], dict) and isinstance(value, dict):
                          # Simple nested merge for one level (e.g., gains, traj_params)
                          merged_config[key].update(value)
                     else:
                         print(f"Warning: Type mismatch for key '{key}' in config. Using default.")

                 else:
                     print(f"Warning: Unknown key '{key}' in config file. Ignored.")
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
    # Convert numpy arrays in config_data to lists for JSON compatibility
    def convert_numpy_to_list(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, dict):
            return {k: convert_numpy_to_list(v) for k, v in item.items()}
        if isinstance(item, list):
            return [convert_numpy_to_list(i) for i in item]
        return item

    config_to_save = convert_numpy_to_list(config_data)

    try:
        with open(filepath, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        print(f"Saved configuration to {filepath}")
    except Exception as e:
        print(f"Error saving config: {e}")

# ---- Simulation Setup ----
DT = 0.02
ANIM_INTERVAL = 20 # ms -> ~50 FPS

# Load initial configuration
current_config = load_config(CONFIG_FILE)

# Initial Drone State & Parameters from config
drone_params = {
    'initial_pos': np.array(current_config['initial_pos']),
    'initial_vel': np.array(current_config['initial_vel']),
    'initial_att': np.array(current_config['initial_att']),
    'initial_omega': np.array(current_config['initial_omega']),
    'mass': current_config['mass'],
    'inertia_diag': np.array(current_config['inertia_diag']),
    'g': current_config['g'],
    'max_thrust_per_prop': current_config['max_thrust_per_prop'],
    'num_propellers': current_config['num_propellers']
}
drone = Drone(**drone_params)

# Controller setup from config
controller = TrajectoryPDController(drone.mass, drone.g,
                                  current_config['controller_gains'],
                                  current_config['trajectory_params'])

# Simulation state
sim_time = 0.0
history_pos = [drone.pos.copy()]
history_target = [controller.target_pos.copy()] # History for target position
history_time = [sim_time]
max_history = 200
current_trajectory_mode = current_config.get('selected_trajectory', 'Hover') # Get from config

# ---- Plotting Setup ----
fig = plt.figure(figsize=(14, 9)) # Adjusted figure size
gs = fig.add_gridspec(2, 2)

# 3D Isometric View
ax3d = fig.add_subplot(gs[0, 0], projection='3d')
ax3d.set_title("Isometric View")
ax3d.set_xlabel("X (m)"), ax3d.set_ylabel("Y (m)"), ax3d.set_zlabel("Z (m)")
line3d, = ax3d.plot([], [], [], 'b-', label="Trajectory")
point3d, = ax3d.plot([], [], [], 'ro', markersize=5, label="Drone")
target3d, = ax3d.plot([], [], [], 'gx', markersize=7, label="Target") # Target marker
# Adjust limits based on typical trajectories
ax3d.set_xlim(-6, 6), ax3d.set_ylim(-6, 6), ax3d.set_zlim(0, 10)
ax3d.legend(fontsize='small')
ax3d.grid(True)

# XY View
ax_xy = fig.add_subplot(gs[0, 1])
ax_xy.set_title("XY View (Top Down)")
ax_xy.set_xlabel("X (m)"), ax_xy.set_ylabel("Y (m)")
line_xy, = ax_xy.plot([], [], 'b-')
point_xy, = ax_xy.plot([], [], 'ro', markersize=5)
target_xy, = ax_xy.plot([], [], 'gx', markersize=7) # Target marker
ax_xy.set_xlim(-6, 6), ax_xy.set_ylim(-6, 6)
ax_xy.set_aspect('equal', adjustable='box'), ax_xy.grid(True)

# YZ View
ax_yz = fig.add_subplot(gs[1, 0])
ax_yz.set_title("YZ View (Side)")
ax_yz.set_xlabel("Y (m)"), ax_yz.set_ylabel("Z (m)")
line_yz, = ax_yz.plot([], [], 'b-')
point_yz, = ax_yz.plot([], [], 'ro', markersize=5)
target_yz, = ax_yz.plot([], [], 'gx', markersize=7) # Target marker
ax_yz.set_xlim(-6, 6), ax_yz.set_ylim(0, 10)
# ax_yz.set_aspect('equal', adjustable='box') # Aspect ratio might make bobbing hard to see
ax_yz.grid(True)

# XZ View
ax_xz = fig.add_subplot(gs[1, 1])
ax_xz.set_title("XZ View (Front)")
ax_xz.set_xlabel("X (m)"), ax_xz.set_ylabel("Z (m)")
line_xz, = ax_xz.plot([], [], 'b-')
point_xz, = ax_xz.plot([], [], 'ro', markersize=5)
target_xz, = ax_xz.plot([], [], 'gx', markersize=7) # Target marker
ax_xz.set_xlim(-6, 6), ax_xz.set_ylim(0, 10)
# ax_xz.set_aspect('equal', adjustable='box')
ax_xz.grid(True)

plt.tight_layout(rect=[0, 0.15, 1, 0.95]) # Adjust layout for more widgets

# ---- GUI Widgets ----
axcolor = 'lightgoldenrodyellow'
widget_x_start = 0.1
widget_y_start = 0.12
widget_height = 0.018
widget_spacing = 0.005
slider_width = 0.55
button_width = 0.08
radio_width = 0.15 # Width for radio buttons area

# Sliders (Position adjusted)
ax_grav = plt.axes([widget_x_start, widget_y_start - 1*(widget_height+widget_spacing), slider_width, widget_height], facecolor=axcolor)
slider_grav = Slider(ax_grav, 'Gravity', 0.0, 20.0, valinit=drone.g, valstep=0.1)

ax_thrust = plt.axes([widget_x_start, widget_y_start - 2*(widget_height+widget_spacing), slider_width, widget_height], facecolor=axcolor)
slider_thrust = Slider(ax_thrust, 'Max Thrust', 1.0, 40.0, valinit=drone.max_total_thrust, valstep=0.5) # Increased range

ax_prop = plt.axes([widget_x_start, widget_y_start - 3*(widget_height+widget_spacing), slider_width, widget_height], facecolor=axcolor)
slider_prop = Slider(ax_prop, 'Num Props', 1, 12, valinit=drone.num_propellers, valstep=1)

# Inertia Sliders (Grouped)
ax_ixx = plt.axes([widget_x_start, widget_y_start - 4*(widget_height+widget_spacing), slider_width/3 - widget_spacing, widget_height], facecolor=axcolor)
slider_ixx = Slider(ax_ixx, 'Ixx', 0.001, 0.1, valinit=drone.I[0,0], valfmt='%0.3f')
ax_iyy = plt.axes([widget_x_start + slider_width/3, widget_y_start - 4*(widget_height+widget_spacing), slider_width/3 - widget_spacing, widget_height], facecolor=axcolor)
slider_iyy = Slider(ax_iyy, 'Iyy', 0.001, 0.1, valinit=drone.I[1,1], valfmt='%0.3f')
ax_izz = plt.axes([widget_x_start + 2*slider_width/3, widget_y_start - 4*(widget_height+widget_spacing), slider_width/3 - widget_spacing, widget_height], facecolor=axcolor)
slider_izz = Slider(ax_izz, 'Izz', 0.001, 0.1, valinit=drone.I[2,2], valfmt='%0.3f')

# Buttons (Reset, Load, Save) - Positioned to the right
button_x_start = widget_x_start + slider_width + 0.05
ax_reset = plt.axes([button_x_start, widget_y_start - 1*(widget_height+widget_spacing), button_width, widget_height])
button_reset = Button(ax_reset, 'Reset', color=axcolor, hovercolor='0.975')

ax_load = plt.axes([button_x_start, widget_y_start - 2*(widget_height+widget_spacing), button_width, widget_height])
button_load = Button(ax_load, 'Load Cfg', color=axcolor, hovercolor='0.975')

ax_save = plt.axes([button_x_start, widget_y_start - 3*(widget_height+widget_spacing), button_width, widget_height])
button_save = Button(ax_save, 'Save Cfg', color=axcolor, hovercolor='0.975')

# Radio Buttons for Trajectory Selection
ax_radio = plt.axes([button_x_start + button_width + 0.02, widget_y_start - 4.5*(widget_height+widget_spacing), radio_width, 4.5*widget_height], facecolor=axcolor) # Taller area
radio_traj = RadioButtons(ax_radio, ('Hover', 'Circle', 'Bobbing', 'Spiral'), active=list(DEFAULT_CONFIG['trajectory_params'].keys()).index(current_trajectory_mode.lower().replace(' ', '_')+'_pos') if current_trajectory_mode=='Hover' else ['hover','circle','bobbing','spiral'].index(current_trajectory_mode.lower()) ) # Find active index
radio_traj.activecolor = 'lightblue'
radio_traj.labels # Access labels if needed

def update_gui_sliders(config):
    """Updates slider values from a config dictionary"""
    slider_grav.set_val(config['g'])
    # Find diagonal inertia if full matrix is stored, otherwise use diag directly
    inertia_diag = config.get('inertia_diag', np.diag(config.get('inertia_matrix', drone.I)).tolist())
    slider_ixx.set_val(inertia_diag[0])
    slider_iyy.set_val(inertia_diag[1])
    slider_izz.set_val(inertia_diag[2])
    # Calculate max total thrust from config
    max_total_t = config['num_propellers'] * config['max_thrust_per_prop']
    slider_thrust.set_val(max_total_t)
    slider_prop.set_val(config['num_propellers'])
    # Update radio button (find active index carefully)
    traj_options = [label.get_text() for label in radio_traj.labels]
    try:
        active_index = traj_options.index(config.get('selected_trajectory', 'Hover'))
        radio_traj.set_active(active_index)
    except ValueError:
        print(f"Warning: Saved trajectory '{config.get('selected_trajectory')}' not found in options. Defaulting.")
        radio_traj.set_active(0) # Default to first option (Hover)


def update_params(val):
    """Callback for sliders."""
    inertia_diag = [slider_ixx.val, slider_iyy.val, slider_izz.val]
    drone_updated = drone.set_params(g=slider_grav.val,
                                     inertia_diag=inertia_diag,
                                     max_thrust=slider_thrust.val,
                                     num_prop=slider_prop.val)

    # Also update controller's view of drone params if they changed
    if drone_updated:
        controller.update_drone_params(drone.mass, drone.g)

# Connect slider update function
slider_grav.on_changed(update_params)
slider_thrust.on_changed(update_params)
slider_prop.on_changed(update_params)
slider_ixx.on_changed(update_params)
slider_iyy.on_changed(update_params)
slider_izz.on_changed(update_params)

def reset_sim(event):
    """Callback for reset button."""
    global sim_time, history_pos, history_target, history_time
    drone.reset()
    # Controller target needs to be reset based on current mode at time 0
    controller.update_target(0.0, current_trajectory_mode)
    controller.update_drone_params(drone.mass, drone.g) # Ensure controller has latest drone params
    sim_time = 0.0
    history_pos = [drone.pos.copy()]
    history_target = [controller.target_pos.copy()]
    history_time = [sim_time]

    # Clear plot lines and points
    line3d.set_data_3d([], [], [])
    point3d.set_data_3d([], [], [])
    target3d.set_data_3d([], [], [])
    line_xy.set_data([], []), point_xy.set_data([], []), target_xy.set_data([], [])
    line_yz.set_data([], []), point_yz.set_data([], []), target_yz.set_data([], [])
    line_xz.set_data([], []), point_xz.set_data([], []), target_xz.set_data([], [])

    # Redraw necessary parts
    fig.canvas.draw_idle()

button_reset.on_clicked(reset_sim)

def load_sim_config(event):
    """Callback for Load Config button."""
    global current_config, current_trajectory_mode
    loaded_config = load_config(CONFIG_FILE)
    current_config = loaded_config # Update global config state

    # Update Drone parameters
    drone_params = {
        'initial_pos': np.array(current_config['initial_pos']),
        'initial_vel': np.array(current_config['initial_vel']),
        'initial_att': np.array(current_config['initial_att']),
        'initial_omega': np.array(current_config['initial_omega']),
        'mass': current_config['mass'],
        'inertia_diag': np.array(current_config['inertia_diag']),
        'g': current_config['g'],
        'max_thrust_per_prop': current_config['max_thrust_per_prop'],
        'num_propellers': current_config['num_propellers']
    }
    # Re-initialize drone with new parameters (preserves class instance)
    drone.__init__(**drone_params)

    # Update Controller parameters
    controller.update_drone_params(drone.mass, drone.g)
    controller.update_gains(current_config['controller_gains'])
    controller.update_trajectory_params(current_config['trajectory_params'])

    # Update GUI elements to reflect loaded config
    update_gui_sliders(current_config)

    # Set the trajectory mode based on loaded config
    current_trajectory_mode = current_config.get('selected_trajectory', 'Hover')
    # Find the index for RadioButtons based on the loaded mode name
    traj_options = [label.get_text() for label in radio_traj.labels]
    try:
        active_index = traj_options.index(current_trajectory_mode)
        radio_traj.set_active(active_index) # This triggers the radio button callback too
    except ValueError:
        radio_traj.set_active(0) # Default if not found
        current_trajectory_mode = traj_options[0]


    # Reset the simulation to apply changes cleanly
    reset_sim(None) # Pass None as event object

button_load.on_clicked(load_sim_config)

def save_sim_config(event):
    """Callback for Save Config button."""
    # Gather current settings from GUI and simulation objects
    config_to_save = {
        # Get initial conditions from the *drone's* initial state storage
        'initial_pos': drone.initial_pos.tolist(),
        'initial_vel': drone.initial_vel.tolist(),
        'initial_att': drone.initial_att.tolist(),
        'initial_omega': drone.initial_omega.tolist(),
        # Get current parameters from the drone object
        'mass': drone.mass,
        'inertia_diag': drone.I.diagonal().tolist(), # Save diagonal
        'g': drone.g,
        'max_thrust_per_prop': drone.max_thrust_per_prop,
        'num_propellers': drone.num_propellers,
        # Get controller gains (assuming they haven't changed dynamically here, but could)
        'controller_gains': {
             'Kp_pos_diag': np.diag(controller.Kp_pos).tolist(),
             'Kd_pos_diag': np.diag(controller.Kd_pos).tolist(),
             'Kp_att_diag': np.diag(controller.Kp_att).tolist(),
             'Kd_att_diag': np.diag(controller.Kd_att).tolist()
        },
        # Get trajectory parameters
        'trajectory_params': controller.traj_params,
         # Get selected trajectory from the radio button's active label
        'selected_trajectory': radio_traj.labels[radio_traj.value_selected.find('x')].get_text() # Simplified way to get text of selected label
    }
    save_config(CONFIG_FILE, config_to_save)

button_save.on_clicked(save_sim_config)


def on_trajectory_select(label):
    """Callback for trajectory radio buttons."""
    global current_trajectory_mode
    current_trajectory_mode = label
    # Update the controller's target immediately for time=sim_time (or time=0 if reset)
    # This prevents a jump if changing trajectories mid-flight
    controller.update_target(sim_time, current_trajectory_mode)
    # No reset needed unless desired when changing modes
    # reset_sim(None) # Optionally reset when changing trajectory
    # Update the config dictionary state (optional, saved on 'Save Cfg')
    current_config['selected_trajectory'] = label

radio_traj.on_clicked(on_trajectory_select)


# ---- Animation Function ----
def animate(frame):
    global sim_time, history_pos, history_target, history_time

    # --- Update Target ---
    # Controller calculates the target position based on time and selected mode
    controller.update_target(sim_time, current_trajectory_mode)

    # --- Control Input ---
    # Uses the trajectory controller now
    total_thrust, torques = controller.calculate_control(
        drone.pos, drone.vel, drone.att_euler, drone.omega, drone.R # Pass rotation matrix
    )

    # --- Perturbation Input (Placeholder) ---
    perturb = None
    # Add perturbations here if needed

    # --- Update Drone State ---
    drone.update(DT, total_thrust, torques, perturbations=perturb)
    sim_time += DT

    # --- Store History ---
    history_pos.append(drone.pos.copy())
    history_target.append(controller.target_pos.copy())
    history_time.append(sim_time)
    if len(history_pos) > max_history:
        history_pos.pop(0)
        history_target.pop(0)
        history_time.pop(0)

    # --- Update Plots ---
    hist_np = np.array(history_pos)
    target_pos_current = controller.target_pos

    # 3D Plot
    line3d.set_data(hist_np[:, 0], hist_np[:, 1])
    line3d.set_3d_properties(hist_np[:, 2])
    point3d.set_data([drone.pos[0]], [drone.pos[1]])
    point3d.set_3d_properties([drone.pos[2]])
    target3d.set_data([target_pos_current[0]], [target_pos_current[1]])
    target3d.set_3d_properties([target_pos_current[2]])

    # XY Plot
    line_xy.set_data(hist_np[:, 0], hist_np[:, 1])
    point_xy.set_data([drone.pos[0]], [drone.pos[1]])
    target_xy.set_data([target_pos_current[0]], [target_pos_current[1]])

    # YZ Plot
    line_yz.set_data(hist_np[:, 1], hist_np[:, 2])
    point_yz.set_data([drone.pos[1]], [drone.pos[2]])
    target_yz.set_data([target_pos_current[1]], [target_pos_current[2]])

    # XZ Plot
    line_xz.set_data(hist_np[:, 0], hist_np[:, 2])
    point_xz.set_data([drone.pos[0]], [drone.pos[2]])
    target_xz.set_data([target_pos_current[0]], [target_pos_current[2]])

    # Return list of artists changed
    return (line3d, point3d, target3d,
            line_xy, point_xy, target_xy,
            line_yz, point_yz, target_yz,
            line_xz, point_xz, target_xz)

# ---- Run Animation ----
ani = animation.FuncAnimation(fig, animate, frames=None, interval=ANIM_INTERVAL, blit=True, repeat=False)

plt.suptitle("Drone Simulation with Configuration and Trajectories")
plt.show()