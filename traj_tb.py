import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

class TrajectoryTensorboardCallback(BaseCallback):
    """
    Every other 100th episode we buffer one trajectory.
    Every 3000 episodes we log three overlaid plots (XY, XZ, YZ)
    covering the last 3000 episodes, then reset the buffer.
    """
    def __init__(self, verbose=0, record_interval: int = 100, block_size: int = 3000):
        super().__init__(verbose)
        self.record_interval = record_interval
        self.block_size = block_size
        self.positions = []
        self.episode_count = 0
        self.buffered_trajs = []

    def _on_training_start(self) -> None:
        self.positions.clear()
        self.episode_count = 0
        self.buffered_trajs.clear()

    def _get_tb_writer(self):
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                return fmt.writer
        raise RuntimeError("No TensorBoardOutputFormat found in logger.")

    def _on_step(self) -> bool:
        # 1) record current position
        pos = self.training_env.get_attr('pos')[0]
        self.positions.append(np.array(pos))

        # 2) check for end of episode
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_count += 1

            # drop the last appended pos (start of next episode)
            traj = np.array(self.positions[:-1])

            # buffer every other 100th episode: 100, 300, 500, …
            if self.episode_count % self.record_interval == 0 and ((self.episode_count // self.record_interval) % 2 == 1):
                self.buffered_trajs.append(traj)

            # every 3000 episodes, log all buffered and reset
            if self.episode_count % self.block_size == 0 and self.buffered_trajs:
                writer = self._get_tb_writer()
                planes = [(0, 1, 'Overlay_XY'),
                          (0, 2, 'Overlay_XZ'),
                          (1, 2, 'Overlay_YZ')]
                block_idx = self.episode_count // self.block_size
                start_ep = (block_idx - 1) * self.block_size + 1
                end_ep = block_idx * self.block_size
                for i, j, tag in planes:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    for run_i, t in enumerate(self.buffered_trajs):
                        ep_num = (2 * run_i + 1) * self.record_interval
                        ax.plot(t[:, i], t[:, j], label=f"ep {ep_num}")
                    ax.set_xlabel(['X', 'Y', 'Z'][i])
                    ax.set_ylabel(['X', 'Y', 'Z'][j])
                    ax.set_title(f"Trajectories {tag} (eps {start_ep}-{end_ep})")
                    writer.add_figure(f"Trajectory/{tag}_block{block_idx}", fig, self.num_timesteps)
                    plt.close(fig)

                self.buffered_trajs.clear()

            # reset for next episode
            self.positions.clear()

        return True