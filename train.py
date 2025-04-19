from stable_baselines3 import PPO
from traj_tb import TrajectoryTensorboardCallback as CB
from drone import DroneGymEnv

model = PPO(
    "MlpPolicy",
    DroneGymEnv(),
    verbose=1,
    tensorboard_log="./tensorboard/"        # must still point somewhere
)
cb = CB()  # no writer argument any more

model.learn(
    total_timesteps=int(1e6),
    tb_log_name="drone_runs",
    callback=cb
)
