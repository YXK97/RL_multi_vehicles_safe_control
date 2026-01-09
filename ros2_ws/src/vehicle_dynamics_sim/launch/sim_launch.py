from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # 声明launch命令行参数（支持外部传入，必须传入path）
    declare_path_arg = DeclareLaunchArgument(
        'path',
        description='Required path parameter (core input for both nodes)'
    )

    # 可选：声明其他需要从命令行传入的参数（如--seed、--debug）
    declare_num_agents_arg = DeclareLaunchArgument('num_agents', default_value='0',
        description='Number of agents, default is None (mapped from ROS2 default 0)')
    declare_env_arg = DeclareLaunchArgument('env', default_value='',
        description='Name of the environment, default is None (mapped from ROS2 default \'\')')
    declare_full_observation_arg = DeclareLaunchArgument('full_observation', default_value='false')
    declare_cpu_arg = DeclareLaunchArgument('cpu', default_value='false')
    declare_max_step_arg = DeclareLaunchArgument('max_step', default_value='0',
        description='Maximum simulation steps of the environment, default is None (mapped from ROS2 default 0)')
    declare_seed_arg = DeclareLaunchArgument('seed', default_value='1234')
    declare_debug_arg = DeclareLaunchArgument('debug', default_value='false')
    declare_area_size_arg = DeclareLaunchArgument('area_size', default_value='')
    declare_from_iter_arg = DeclareLaunchArgument('from_iter', default_value='0',
        description='Checkpoint index of the network, default is None (mapped from ROS2 default 0)')
    declare_stochastic_arg = DeclareLaunchArgument('stochastic', default_value='false')

    # 定义所有参数（整合两个节点，默认值与原argparse一致）
    # 公共参数（传递给两个节点）
    common_params = {
        'path': LaunchConfiguration('path'),
        'num_agents': LaunchConfiguration('num_agents'),
        'env': LaunchConfiguration('env'),
        'full_observation': LaunchConfiguration('full_observation'),
        'cpu': LaunchConfiguration('cpu'),
        'max_step': LaunchConfiguration('max_step'),
        'seed': LaunchConfiguration('seed'),
        'debug': LaunchConfiguration('debug'),
        'area_size': LaunchConfiguration('area_size')
    }
    # action节点独有参数（在公共参数基础上扩展）
    action_params = {
        **common_params,  # 继承公共参数
        'from_iter': LaunchConfiguration('from_iter'),
        'stochastic': LaunchConfiguration('stochastic'),
    }

    # 定义两个节点（传递对应参数）
    # env节点
    env_node = Node(
        package='vehicle_dynamics_sim',
        executable='start_env_node',
        name='start_env_node',
        output='screen',
        parameters=[common_params]  # 传递公共参数
    )
    # action节点
    action_node = Node(
        package='vehicle_dynamics_sim',
        executable='start_action_node',
        name='start_action_node',
        output='screen',
        parameters=[action_params]  # 传递整合后的所有参数
    )

    # 节点联动逻辑（保持原有启停联动）
    start_action_after_env = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=env_node,
            on_start=[action_node]
        )
    )
    stop_action_after_env = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=env_node,
            on_exit=[
                Node(
                    package='ros2cli',
                    executable='node',
                    arguments=['kill', '/start_action_node']
                )
            ]
        )
    )

    return LaunchDescription([
        # 声明命令行参数
        declare_path_arg,
        declare_num_agents_arg,
        declare_env_arg,
        declare_full_observation_arg,
        declare_cpu_arg,
        declare_max_step_arg,
        declare_seed_arg,
        declare_debug_arg,
        declare_area_size_arg,
        declare_from_iter_arg,
        declare_stochastic_arg,
        # 节点与联动逻辑
        env_node,
        start_action_after_env,
        stop_action_after_env
    ])