# 🐢 Training Turtle Locomotion using Reinforcement Learning in Mujoco

A custom gymnasium environment for training turtle locomotion using reinforcement learning in the Mujoco simulator. The environment has been set up for a custom Turtle robot, however, it can be easily extended to train other robots as well.

There are two MJCF models provided for the Turtle robot. One tuned for position control with a proportional controller, and one model which directly takes in torque values for end-to-end training.

### 🦾 Trained Model with Motor Torque Actions
https://github.com/nimazareian/quadruped-rl-locomotion/assets/28585597/262b7812-0b8f-4758-aedd-a429f743fb69

### 🎯 Trained Model with Position Actions and a Proportional Controller
https://github.com/nimazareian/quadruped-rl-locomotion/assets/28585597/f0eddef1-7bc4-4d7a-adc5-35e630ced5d4

## 📦 Setup
```bash
python -m pip install -r requirements.txt
```

## 🚀 Train
```bash
python turtle_train.py --run train
```

## 🖥️ Displaying Trained Models 

```bash
python turtle_train.py --run test --model_path <path to model zip file>
```
For example, to run a pretrained model which outputs motor torques and has the robot desired velocity set to <x=1, y=0>, you can run:
```bash
python turtle_train.py --run test --model_path ./models/2026-05-16_23-42-36/final_model.zip
```

## 🙏 Acknowledgements

This work was built upon the excellent foundation of the [quadruped-rl-locomotion](https://github.com/nimazareian/quadruped-rl-locomotion) repository. Many thanks to the original authors for making it so easy to port and adapt the environment to train my Turtle robot! 🐢✨
