#!/bin/bash
# run_sim.sh - 专属启动脚本，隔离conda与ROS2环境

# 1. 激活conda环境（defmarl）
source /home/yxk-vtd/miniforge3/etc/profile.d/conda.sh  # 替换为你的conda初始化路径
conda activate defmarl

# 2. 局部配置环境变量（仅对当前脚本生效，不污染全局）
# 告诉ROS2节点用conda的Python3.10（仅影响节点，不影响ros2命令）
export ROS_PYTHON_VERSION=3.10
# 仅将conda的site-packages加入PYTHONPATH，但放在系统ROS2库之后（避免覆盖系统ros2命令的库）
export PYTHONPATH=/opt/ros/galactic/lib/python3.8/site-packages:$CONDA_PREFIX/lib/python3.10/site-packages:$PYTHONPATH

# 3. 编译你的ROS2功能包（如果需要）
cd ~/RL_multi_vehicles_safe_control/ros2_ws
colcon build --packages-select vehicle_dynamics_sim
source install/setup.bash

# 4. 启动你的仿真节点（替换为你的launch文件）
ros2 launch vehicle_dynamics_sim sim_launch.py

# 5. 退出时关闭conda环境
conda deactivate
