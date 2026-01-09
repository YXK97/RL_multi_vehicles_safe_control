#!/bin/bash
# 专属运行脚本：彻底隔离conda与ROS2系统环境，直接指定Python解释器

# 1. 仅加载ROS2系统环境和当前工作空间（不激活conda）
unset PYTHONPATH
unset CONDA_PREFIX
source /opt/ros/galactic/setup.bash
source /home/yxk-vtd/RL_multi_vehicles_safe_control/ros2_ws/install/setup.bash

# 2. 配置ROS2使用conda的Python 3.10解释器（关键：直接指定路径，不激活conda）
export ROS_PYTHON_VERSION=3.10
export PYTHON_EXECUTABLE=/home/yxk-vtd/miniforge3/envs/defmarl/bin/python3.10
export PYTHONPATH=/home/yxk-vtd/miniforge3/envs/defmarl/lib/python3.10/site-packages:$PYTHONPATH

# 3. 启动仿真launch文件（转发所有命令行参数，核心是--path）
# 用法：./run_sim.sh --path /your/required/path [--num_agents 5]
ros2 launch vehicle_dynamics_sim sim_launch.py "$@"
