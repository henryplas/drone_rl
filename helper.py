import os
import re
from pathlib import Path
from stable_baselines3.common.logger import configure

def make_run_dir(root_dir: str, prefix: str = "drone_runs_") -> str:
    """
    Scans root_dir for existing folders named prefix{n}, picks the next n.
    E.g. if you have drone_runs_1, drone_runs_2, it will return './tensorboard/drone_runs_3'
    """
    os.makedirs(root_dir, exist_ok=True)
    existing = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for name in os.listdir(root_dir):
        m = pattern.match(name)
        if m:
            existing.append(int(m.group(1)))
    next_idx = max(existing, default=0) + 1
    run_dir = os.path.join(root_dir, f"{prefix}{next_idx}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

