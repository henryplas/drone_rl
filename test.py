from drone import DroneGymEnv
from stable_baselines3 import PPO
import time

s = time.time()
env = DroneGymEnv()
model = PPO.load('./dd.zip', env=env)

# start recording to MP4
env.start_record('my_drone_run.gif', dpi=200, fps=20)

obs = env.reset()
for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()    # draws + grabs frame
    if done:
        obs = env.reset()

# finish & save
env.stop_record()

print(time.time()-s)