from drone import DroneGymEnv
from stable_baselines3 import PPO

# Create the Gym environment.
env = DroneGymEnv()

# Instantiate the PPO agent using an MLP policy.
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

# Train for a total of 100,000 timesteps.
model.learn(total_timesteps=10e6,  tb_log_name="first_run")

# Save the trained model.
model.save("ppo_drone")

# Test the trained model.
# obs = env.reset()
# for i in range(300):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()