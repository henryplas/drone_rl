from stable_baselines3 import PPO
from traj_tb import TrajectoryTensorboardCallback as CB
from drone import DroneGymEnv
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure
from helper import make_run_dir
import torch

file_path = Path('./dd.zip')
total_steps = 2e6
if file_path.exists():
    n_envs  = 1
    n_steps = 2048 // n_envs   # keeps 2048 total transitions per update
    batch_size = 64            # same as your single‑env setting

    # each fn must _call_ the constructor, not return the class
    env_fns = [lambda: DroneGymEnv() for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env) 
    
    model = PPO.load(
    str(file_path), 
    vec_env,
    n_steps=n_steps,
    batch_size=batch_size,
    learning_rate=3e-4,
    verbose=1,
    device="cpu",
    )
    print('hi')
else:
    env_fns = [lambda: DroneGymEnv() for _ in range(1)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env) 
    model = PPO(
        "MlpPolicy",
        verbose=1,
        # tensorboard_log="./tensorboard/",
        # str(file_path),
        env=vec_env,
        device="cpu"
    )

import time
start = time.time()
obs = vec_env.reset()
actions, _ = model.predict(obs)
vec_env.step(actions)
model.policy.forward(torch.as_tensor(obs).to(model.device))
# ...and model.policy.evaluate_actions() / optimizer.step()
print("Wall‑clock per iter:", time.time() - start)


tensorboard_root = "./tensorboard"
tb_log_dir = make_run_dir(tensorboard_root, prefix="drone_runs_")
new_logger = configure(tb_log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)
model.verbose = 1

cb = CB()

model.learn(
    total_timesteps=int(total_steps),
    tb_log_name="drone_runs",
    callback=cb,
    # reset_num_timesteps=False,
)

model.save("ppo_drone_rel_obs_pos_reward")