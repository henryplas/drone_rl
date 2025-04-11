import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation  # Easier rotation handling

# ---- Drone Physics and State Class ----
class Drone:
    def __init__(self,
                 initial_pos=np.array([0.0, 0.0, 5.0]),
                 initial_vel=np.array([0.0, 0.0, 0.0]),
                 initial_att=np.array([0.0, 0.0, 0.0]),  # Euler angles: roll, pitch, yaw (rad)
                 initial_omega=np.array([0.0, 0.0, 0.0]), # Angular velocity: p, q, r (rad/s) in body frame
                 mass=1.0, # kg
                 inertia_matrix=np.diag([0.01, 0.01, 0.02]), # Ixx, Iyy, Izz kg*m^2 (body frame)
                 g=9.81, # m/s^2
                 max_thrust_per_prop=2.0, # N
                 num_propellers=4 # Affects total max thrust and control allocation (simplified here)
                 ):
        # --- Initial State ---
        self.initial_pos = initial_pos.copy()
        self.initial_vel = initial_vel.copy()
        self.initial_att = initial_att.copy()
        self.initial_omega = initial_omega.copy()

        # --- Current State ---
        self.pos = self.initial_pos.copy()         # [x, y, z] inertial frame
        self.vel = self.initial_vel.copy()         # [vx, vy, vz] inertial frame
        self.att_euler = self.initial_att.copy()   # [roll, pitch, yaw] inertial frame angles
        self.omega = self.initial_omega.copy()     # [p, q, r] body frame angular velocity

        # --- Parameters ---
        self.mass = mass
        self.I = inertia_matrix                     # Inertia Tensor (body frame)
        self.I_inv = np.linalg.inv(self.I)          # Inverse Inertia Tensor
        self.g = g
        self.num_propellers = num_propellers
        self.max_thrust_per_prop = max_thrust_per_prop
        self.max_total_thrust = self.num_propellers * self.max_thrust_per_prop

        # --- Internal ---
        self._update_rotation_matrix()              # Initialize rotation matrix

    def _update_rotation_matrix(self):
        # Update the rotation matrix from body frame to inertial frame
        # Using scipy's Rotation for clarity and robustness vs manual Euler matrix
        # Sequence 'ZYX' corresponds to yaw, pitch, roll
        self.R = Rotation.from_euler('ZYX', self.att_euler[::-1]).as_matrix() # Note: scipy uses ZYX order for yaw,pitch,roll

    def _euler_rate_to_omega(self, euler_rates):
        # Kinematic relationship: Convert angular velocity (omega) from body frame to euler angle rates
        # This is the *inverse* of the standard matrix W found in textbooks
        roll, pitch, yaw = self.att_euler
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        tp = np.tan(pitch)

        # Avoid singularity at pitch = +/- 90 deg
        if abs(cp) < 1e-6:
            cp = np.sign(cp) * 1e-6 # Avoid division by zero

        W_inv = np.array([
            [1, sr * tp, cr * tp],
            [0, cr, -sr],
            [0, sr / cp, cr / cp]
        ])
        return W_inv @ euler_rates # This calculation is not directly needed for update, but useful for reference

    def _omega_to_euler_rate(self):
        # Convert body frame angular velocity (p, q, r) to Euler angle rates (d(roll)/dt, d(pitch)/dt, d(yaw)/dt)
        roll, pitch, yaw = self.att_euler
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        tp = np.tan(pitch)

        # Avoid singularity at pitch = +/- 90 deg
        if abs(cp) < 1e-6:
            cp = np.sign(cp) * 1e-6 # Avoid division by zero

        W = np.array([
            [1, sr * tp, cr * tp],
            [0, cr,      -sr],
            [0, sr / cp, cr / cp]
        ])
        return W @ self.omega

    def reset(self):
        """Resets the drone state to initial conditions."""
        self.pos = self.initial_pos.copy()
        self.vel = self.initial_vel.copy()
        self.att_euler = self.initial_att.copy()
        self.omega = self.initial_omega.copy()
        self._update_rotation_matrix()

    def set_params(self, g=None, inertia_diag=None, max_thrust=None):
        """Allows updating parameters dynamically."""
        if g is not None:
            self.g = g
        if inertia_diag is not None:
            # Ensure it's a 3-element array/list
            if len(inertia_diag) == 3:
                self.I = np.diag(inertia_diag)
                # Handle potential non-invertibility if values are zero/negative
                try:
                    self.I_inv = np.linalg.inv(self.I)
                except np.linalg.LinAlgError:
                    print("Warning: Inertia matrix became singular. Using pseudo-inverse.")
                    self.I_inv = np.linalg.pinv(self.I)
            else:
                print("Warning: Inertia requires 3 diagonal elements [Ixx, Iyy, Izz]. Not updated.")
        if max_thrust is not None:
             # This assumes changing total thrust proportionally changes per-prop thrust
             self.max_total_thrust = max_thrust
             if self.num_propellers > 0:
                 self.max_thrust_per_prop = max_thrust / self.num_propellers
             else:
                 self.max_thrust_per_prop = 0


    def update(self, dt, total_thrust, torques, perturbations=None):
        """
        Updates the drone state based on inputs and physics.

        Args:
            dt (float): Time step (s)
            total_thrust (float): Total thrust force (N) along the drone's body z-axis.
                                 Should be bounded [0, max_total_thrust].
            torques (np.array): Control torques [tau_x, tau_y, tau_z] (N*m) in the body frame.
            perturbations (dict, optional): {'force': [fx, fy, fz], 'torque': [tx, ty, tz]}
                                           Forces are in inertial frame, torques in body frame.
        """
        # --- Clamp Inputs ---
        total_thrust = np.clip(total_thrust, 0, self.max_total_thrust)

        # --- Forces (Inertial Frame) ---
        # Gravity
        F_gravity = np.array([0, 0, -self.mass * self.g])

        # Thrust (Body frame to Inertial Frame)
        F_thrust_body = np.array([0, 0, total_thrust])
        F_thrust_inertial = self.R @ F_thrust_body

        # Perturbations
        F_perturb_inertial = np.zeros(3)
        if perturbations and 'force' in perturbations:
            F_perturb_inertial = np.array(perturbations['force'])

        # Total Force
        F_total = F_gravity + F_thrust_inertial + F_perturb_inertial

        # --- Linear Dynamics (Newton's Second Law) ---
        linear_accel = F_total / self.mass
        self.vel += linear_accel * dt
        self.pos += self.vel * dt

        # Prevent falling through floor (simple ground collision)
        if self.pos[2] < 0:
            self.pos[2] = 0
            self.vel[2] = max(0, self.vel[2]) # Stop downward velocity, allow bounce up

        # --- Torques (Body Frame) ---
        tau_ctrl = np.array(torques)

        # Perturbations
        tau_perturb_body = np.zeros(3)
        if perturbations and 'torque' in perturbations:
            tau_perturb_body = np.array(perturbations['torque'])

        # Gyroscopic effect (omega x I*omega)
        tau_gyro = np.cross(self.omega, self.I @ self.omega)

        # Total Torque
        tau_total = tau_ctrl - tau_gyro + tau_perturb_body # Note: gyro term is often subtracted

        # --- Rotational Dynamics (Euler's Equations) ---
        angular_accel = self.I_inv @ tau_total
        self.omega += angular_accel * dt

        # --- Update Orientation ---
        # Integrate angular velocity to get new orientation
        # Using Euler angle rates: d(euler)/dt = W(euler) * omega
        euler_rates = self._omega_to_euler_rate()
        self.att_euler += euler_rates * dt

        # Keep yaw between -pi and pi (optional, but good practice)
        # self.att_euler[2] = (self.att_euler[2] + np.pi) % (2 * np.pi) - np.pi

        # Update the rotation matrix for the next step
        self._update_rotation_matrix()

# ---- Simple PD Controller for Hovering ----
class SimplePDController:
    def __init__(self, drone_mass, drone_g):
        self.target_pos = np.array([0.0, 0.0, 5.0])
        self.target_att_euler = np.array([0.0, 0.0, 0.0]) # Target roll=0, pitch=0, yaw=0

        # --- Gains (These need tuning!) ---
        # Position gains
        self.Kp_pos = np.diag([2.0, 2.0, 5.0])  # Proportional gains for x, y, z
        self.Kd_pos = np.diag([2.5, 2.5, 4.0])  # Derivative gains for vx, vy, vz

        # Attitude gains (for roll, pitch control)
        self.Kp_att = np.diag([50.0, 50.0, 20.0]) # Proportional gains for roll, pitch, yaw
        self.Kd_att = np.diag([10.0, 10.0, 5.0])  # Derivative gains for p, q, r

        # Store drone parameters needed for control
        self.mass = drone_mass
        self.g = drone_g

    def update_drone_params(self, mass, g):
        """Update controller parameters if drone parameters change"""
        self.mass = mass
        self.g = g

    def calculate_control(self, current_pos, current_vel, current_att_euler, current_omega):
        # --- Position Control (calculates desired thrust and attitude) ---
        pos_error = self.target_pos - current_pos
        vel_error = -current_vel # Target velocity is zero

        # Desired force in inertial frame (PD control)
        F_desired = self.Kp_pos @ pos_error + self.Kd_pos @ vel_error + np.array([0, 0, self.mass * self.g])

        # Required total thrust (magnitude of desired force projection onto body z-axis)
        # This is a simplification. Ideally, project F_desired onto current body z-axis.
        # For small roll/pitch angles, we can approximate:
        total_thrust = F_desired[2] # Primarily use Z component for thrust magnitude near hover

        # --- Attitude Control (calculates desired torques) ---
        # Calculate desired roll/pitch from desired force XY components
        # (This is a simplified approach, more sophisticated methods exist)
        # Small angle approximation: desired_roll ~ -F_desired_y / total_thrust, desired_pitch ~ F_desired_x / total_thrust
        # Avoid division by zero if thrust is near zero
        if total_thrust > 0.1:
             # Note the sign conventions might need adjustment depending on axis definitions
             desired_roll = np.clip(F_desired[1] / total_thrust, -0.5, 0.5) # Limit desired angles
             desired_pitch = np.clip(-F_desired[0] / total_thrust, -0.5, 0.5)
        else:
             desired_roll = 0
             desired_pitch = 0

        # For now, keep target yaw fixed
        desired_yaw = self.target_att_euler[2]
        target_attitude = np.array([desired_roll, desired_pitch, desired_yaw])

        # Attitude Error
        att_error = target_attitude - current_att_euler
        # Ensure yaw error is wrapped correctly (shortest angle)
        att_error[2] = (att_error[2] + np.pi) % (2 * np.pi) - np.pi

        # Angular Velocity Error (target omega is zero for stabilization)
        omega_error = -current_omega

        # Desired Torques (PD control)
        torques = self.Kp_att @ att_error + self.Kd_att @ omega_error

        return total_thrust, torques


# ---- Simulation Setup ----
DT = 0.02  # Simulation time step (s)
ANIM_INTERVAL = 20 # Animation update interval (ms) -> Target ~50 FPS

# Initial Drone State & Parameters
initial_params = {
    'initial_pos': np.array([0.0, 0.0, 1.0]), # Start lower
    'initial_vel': np.array([0.0, 0.0, 0.0]),
    'initial_att': np.array([0.0, 0.0, 0.0]),
    'initial_omega': np.array([0.0, 0.0, 0.0]),
    'mass': 1.0,
    'inertia_matrix': np.diag([0.005, 0.005, 0.01]), # Smaller inertia
    'g': 9.81,
    'max_thrust_per_prop': 5.0, # More thrust capacity
    'num_propellers': 4
}
drone = Drone(**initial_params)
controller = SimplePDController(drone.mass, drone.g)

# Simulation state
sim_time = 0.0
history_pos = [drone.pos.copy()]
history_time = [sim_time]
max_history = 200 # Number of points to keep in trajectory history

# ---- Plotting Setup ----
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2)

# 3D Isometric View
ax3d = fig.add_subplot(gs[0, 0], projection='3d')
ax3d.set_title("Isometric View")
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")
line3d, = ax3d.plot([], [], [], 'b-', label="Trajectory")
point3d, = ax3d.plot([], [], [], 'ro', markersize=5, label="Drone Position")
ax3d.set_xlim(-5, 5)
ax3d.set_ylim(-5, 5)
ax3d.set_zlim(0, 10)
ax3d.legend()
ax3d.grid(True)

# XY View
ax_xy = fig.add_subplot(gs[0, 1])
ax_xy.set_title("XY View (Top Down)")
ax_xy.set_xlabel("X (m)")
ax_xy.set_ylabel("Y (m)")
line_xy, = ax_xy.plot([], [], 'b-')
point_xy, = ax_xy.plot([], [], 'ro', markersize=5)
ax_xy.set_xlim(-5, 5)
ax_xy.set_ylim(-5, 5)
ax_xy.set_aspect('equal', adjustable='box')
ax_xy.grid(True)

# YZ View
ax_yz = fig.add_subplot(gs[1, 0])
ax_yz.set_title("YZ View (Side)")
ax_yz.set_xlabel("Y (m)")
ax_yz.set_ylabel("Z (m)")
line_yz, = ax_yz.plot([], [], 'b-')
point_yz, = ax_yz.plot([], [], 'ro', markersize=5)
ax_yz.set_xlim(-5, 5)
ax_yz.set_ylim(0, 10)
ax_yz.set_aspect('equal', adjustable='box')
ax_yz.grid(True)

# XZ View
ax_xz = fig.add_subplot(gs[1, 1])
ax_xz.set_title("XZ View (Front)")
ax_xz.set_xlabel("X (m)")
ax_xz.set_ylabel("Z (m)")
line_xz, = ax_xz.plot([], [], 'b-')
point_xz, = ax_xz.plot([], [], 'ro', markersize=5)
ax_xz.set_xlim(-5, 5)
ax_xz.set_ylim(0, 10)
ax_xz.set_aspect('equal', adjustable='box')
ax_xz.grid(True)

plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust layout to make space for widgets

# ---- GUI Widgets ----
axcolor = 'lightgoldenrodyellow'

# Sliders
ax_grav = plt.axes([0.15, 0.06, 0.65, 0.02], facecolor=axcolor)
slider_grav = Slider(ax_grav, 'Gravity', 0.0, 20.0, valinit=drone.g)

ax_thrust = plt.axes([0.15, 0.03, 0.65, 0.02], facecolor=axcolor)
slider_thrust = Slider(ax_thrust, 'Max Thrust', 1.0, drone.num_propellers * 10.0, valinit=drone.max_total_thrust)

# Inertia Sliders (assuming diagonal)
ax_ixx = plt.axes([0.1, 0.01, 0.15, 0.015], facecolor=axcolor)
slider_ixx = Slider(ax_ixx, 'Ixx', 0.001, 0.1, valinit=drone.I[0,0])
ax_iyy = plt.axes([0.3, 0.01, 0.15, 0.015], facecolor=axcolor)
slider_iyy = Slider(ax_iyy, 'Iyy', 0.001, 0.1, valinit=drone.I[1,1])
ax_izz = plt.axes([0.5, 0.01, 0.15, 0.015], facecolor=axcolor)
slider_izz = Slider(ax_izz, 'Izz', 0.001, 0.1, valinit=drone.I[2,2])

# Reset Button
ax_reset = plt.axes([0.85, 0.02, 0.1, 0.04])
button_reset = Button(ax_reset, 'Reset', color=axcolor, hovercolor='0.975')

def update_params(val):
    """Callback for sliders."""
    inertia_diag = [slider_ixx.val, slider_iyy.val, slider_izz.val]
    drone.set_params(g=slider_grav.val,
                     inertia_diag=inertia_diag,
                     max_thrust=slider_thrust.val)
    # Also update controller parameters if they depend on drone params
    controller.update_drone_params(drone.mass, drone.g)

slider_grav.on_changed(update_params)
slider_thrust.on_changed(update_params)
slider_ixx.on_changed(update_params)
slider_iyy.on_changed(update_params)
slider_izz.on_changed(update_params)

def reset_sim(event):
    """Callback for reset button."""
    global sim_time, history_pos, history_time
    drone.reset()
    controller.update_drone_params(drone.mass, drone.g) # Reset controller params too
    sim_time = 0.0
    history_pos = [drone.pos.copy()]
    history_time = [sim_time]
    # Clear plot lines
    line3d.set_data([], [])
    line3d.set_3d_properties([])
    line_xy.set_data([], [])
    line_yz.set_data([], [])
    line_xz.set_data([], [])
    # Reset points
    point3d.set_data([],[])
    point3d.set_3d_properties([])
    point_xy.set_data([],[])
    point_yz.set_data([],[])
    point_xz.set_data([],[])
    # Reset target position display if you add one
    # You might need to redraw the figure if axes limits changed significantly
    fig.canvas.draw_idle()

button_reset.on_clicked(reset_sim)

# ---- Animation Function ----
def animate(frame):
    global sim_time, history_pos, history_time

    # --- Control Input ---
    # Replace this with your RL agent or other controller
    # Currently uses the simple PD hover controller
    total_thrust, torques = controller.calculate_control(
        drone.pos, drone.vel, drone.att_euler, drone.omega
    )

    # --- Perturbation Input (Example) ---
    perturb = None
    # Example: Apply a random force impulse occasionally
    # if np.random.rand() < 0.01: # Apply perturbation ~1% of the time steps
    #      force_perturb = (np.random.rand(3) - 0.5) * 5.0 # Random force [-2.5, 2.5] N
    #      perturb = {'force': force_perturb}
    #      print(f"Applying Force Perturbation: {force_perturb}")

    # --- Update Drone State ---
    drone.update(DT, total_thrust, torques, perturbations=perturb)
    sim_time += DT

    # --- Store History ---
    history_pos.append(drone.pos.copy())
    history_time.append(sim_time)
    if len(history_pos) > max_history:
        history_pos.pop(0)
        history_time.pop(0)

    # --- Update Plots ---
    hist = np.array(history_pos)

    # 3D Plot
    line3d.set_data(hist[:, 0], hist[:, 1])
    line3d.set_3d_properties(hist[:, 2])
    point3d.set_data([drone.pos[0]], [drone.pos[1]])
    point3d.set_3d_properties([drone.pos[2]])

    # XY Plot
    line_xy.set_data(hist[:, 0], hist[:, 1])
    point_xy.set_data([drone.pos[0]], [drone.pos[1]])

    # YZ Plot
    line_yz.set_data(hist[:, 1], hist[:, 2])
    point_yz.set_data([drone.pos[1]], [drone.pos[2]])

    # XZ Plot
    line_xz.set_data(hist[:, 0], hist[:, 2])
    point_xz.set_data([drone.pos[0]], [drone.pos[2]])

    # Dynamically adjust plot limits slightly if needed (optional)
    # Can impact performance
    # ax3d.set_xlim(min(hist[:,0])-1, max(hist[:,0])+1)
    # ... etc for other axes and plots

    # Return list of artists changed (required for blitting)
    return line3d, point3d, line_xy, point_xy, line_yz, point_yz, line_xz, point_xz,

# ---- Run Animation ----
# Use blit=True for potentially smoother animation, but can sometimes cause issues
# If animation glitches, try blit=False
ani = animation.FuncAnimation(fig, animate, frames=None, interval=ANIM_INTERVAL, blit=True, repeat=False)

plt.suptitle("Drone Simulation")
plt.show()
