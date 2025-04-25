# Drone Reinforcement Learning Simulation

## Overview
This repository contains the simulation environment and training scripts for a deep reinforcement learning agent to control a quadcopter/drone in Python. Using OpenAI Gym (or Gymnasium) and Stable-Baselines3, you can train policies (e.g., PPO) to navigate the drone from a start point to a target point in 3D space, with options for video recording and custom callbacks.

## Table of Contents
- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Training](#training)  
  - [Recording Video](#recording-video)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

## Features
- Custom Gym environment (`DroneGymEnv`) with physics-based simulation  
- Support for single- and multi-environment vectorized training  
- Pre-built PPO training and evaluation scripts  
- TensorBoard logging and custom callbacks  
- Video recording of rollouts (MP4/GIF)  

## Prerequisites
- Python 3.8+  

## Installation
1. Clone this repository:  
   ```bash
   git clone https://github.com/henryplas/drone_rl.git
   cd drone_rl
   ```

2. Create a virtual environment and activate it:  
   ```bash
   conda env create -f environment.yaml
   conda activate drone-rl
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
Run the main training script using PPO:  
```bash
python train.py 
```

### Recording Video
To record rollouts to an MP4:  
```bash
python record.py --model-path logs/ppo/model.zip --output video.mp4 --fps 20
```

## Project Structure
```plaintext
drone_rl/
├── drone.py            # Custom Gym environment implementation
├── train.py            # Training entry point
├── test.py             # Video recording utility
├── environment.yaml    # Python dependencies
├── tensorboard/        # TensorBoard and model outputs
└── README.md           # This file
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request:  
1. Fork the repo  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to the branch (`git push origin feature-name`)  
5. Open a pull request  

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or feedback, contact Henry Plaskonos at <henry.plaskonos@gmail.com>. Happy flying!
```
