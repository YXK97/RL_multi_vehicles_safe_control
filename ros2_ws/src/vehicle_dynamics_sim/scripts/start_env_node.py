import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 由于是测试，没有进行数据并行的必要，强制使用第0个GPU
import rclpy
import time
import yaml
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import Optional
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from defmarl.env import make_env
from defmarl.env.mve import MVE, MVEEnvGraphsTuple
from defmarl.utils.utils import parse_jax_array
from vehicle_dynamics_sim.action import AgentControl
from vehicle_dynamics_sim.msg import StateAndEval, SingleObjectState


class EnvNode(Node):
    def __init__(self):
        super().__init__ ('start_env_node')

        # 必填参数（无默认值，后续在launch中强制传入）
        self.declare_parameter('path', '')  # 核心必填参数，无默认值
        # 可选参数（与原argparse默认值一致）
        self.declare_parameter('num_agents', None)
        self.declare_parameter('env', None)
        self.declare_parameter('full_observation', False)  # ROS2参数名不支持连字符，用下划线
        self.declare_parameter('cpu', False)
        self.declare_parameter('max_step', None)
        self.declare_parameter('seed', 1234)
        self.declare_parameter('debug', False)
        self.declare_parameter('area_size', '')  # jax数组先传字符串，再内部解析

        # 取所有参数（封装为类似原args的对象，方便后续调用
        args = self.get_all_parameters()

        # 校验必填参数（path不能为空）
        if not args.path:
            self.get_logger().fatal('Parameter "path" is required! Please set it in launch file.')
            rclpy.shutdown()
            raise SystemExit('Missing required parameter: path')

        n_gpu = jax.local_device_count()
        print(f"> initializing EnvNode {args}")
        print(f"> Using {n_gpu} devices")

        # set up environment variables and seed
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if args.cpu:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
        if args.debug:
            jax.config.update("jax_disable_jit", True)

        # load config
        if args.path is not None:
            with open(os.path.join(args.path, "config.yaml"), "r") as f:
                config = yaml.load(f, Loader=yaml.UnsafeLoader)

        # create environments
        num_agents = config.num_agents if args.num_agents is None else args.num_agents
        env = make_env(
            env_id=config.env if args.env is None else args.env,
            num_agents=num_agents,
            num_obs=config.obs,
            max_step=args.max_step,
            full_observation=args.full_observation,
            area_size=config.area_size if args.area_size is None else args.area_size,
        )
        self.env = env

        # 初始化状态和步数
        np.random.seed(args.seed)
        key_0 = jr.PRNGKey(args.seed)
        self.current_graph = self.env.reset(key_0)
        self.current_step = 0


        # ros通信
        # 状态量发布到 /ros_env/vehicle_state
        self.state_pub = self.create_publisher(StateAndEval, '/ros_env', 0)
        # 创建Action客户端（向ros_action节点请求控制量）
        self.control_client = ActionClient(self, AgentControl, '/ros_action')
        # 定时器（40Hz执行step）
        self.timer = self.create_timer(self.env.dt, self.step_callback)

        # 运行标记
        self.is_running = True
        self.get_logger().info('Env node initialized, max step: %d, freq: %dHz' % (self.env.max_step, 1/self.env.dt))


    def get_all_parameters(self):
        """获取所有ROS2参数，封装为简单对象（模拟argparse的args）"""
        class EnvArgs:
            path:str
            num_agents:Optional[int]
            env:Optional[str]
            full_observation:bool
            cpu:bool
            max_step:Optional[int]
            seed:int
            debug:bool
            area_size:jnp.ndarray
        args = EnvArgs()
        # 赋值所有参数
        args.path = self.get_parameter('path').get_parameter_value().string_value
        args.num_agents = self.get_parameter('num_agents').get_parameter_value().integer_value
        args.env = self.get_parameter('env').get_parameter_value().string_value
        args.full_observation = self.get_parameter('full_observation').get_parameter_value().bool_value
        args.cpu = self.get_parameter('cpu').get_parameter_value().bool_value
        args.max_step = self.get_parameter('max_step').get_parameter_value().integer_value
        args.seed = self.get_parameter('seed').get_parameter_value().integer_value
        args.debug = self.get_parameter('debug').get_parameter_value().bool_value
        # 处理自定义jax数组（解析字符串为jax数组，复用原parse_jax_array）
        area_size_str = self.get_parameter('area_size').get_parameter_value().string_value
        args.area_size = parse_jax_array(area_size_str) if area_size_str else None
        # 处理None值（ROS2参数未指定时，integer_value/string_value会返回0/空字符串，修正为None）
        for param in ['num_agents', 'env', 'max_step']:
            val = getattr(args, param)
            if val == 0 or val == '':
                setattr(args, param, None)
        return args


    def step_callback(self):
        """核心step逻辑：40Hz执行"""
        if not self.is_running or self.current_step >= self.env.max_step:
            self.terminate_sim()
            return

        step_start = time.time()
        # 获取控制量
        ad_action = self.get_control_cmd()  # [acc, delta]
        # 执行动力学step
        next_graph, reward, a_cost, done, info = self.env.step(self.current_graph, ad_action)
        cost = a_cost.max

        # 发布当前时刻环境状态量、控制量、评估量并更新
        self.publish_state(self.current_graph, reward, cost)

        # 同步现实时间：补足时延
        step_elapsed = time.time() - step_start
        if step_elapsed < self.env.dt:
            time.sleep(self.env.dt - step_elapsed)

        # 更新状态
        self.current_step += 1
        self.current_graph = next_graph
        if self.current_step % 10 == 0:  # 每10步打印日志
            self.get_logger().debug(f'Step {self.current_step}/{self.env.max_step}, reward: {reward:.2f}, cost: {cost:.2f}')


    def get_control_cmd(self):
        """向action节点请求控制量，默认[0,0]"""
        default_action = jnp.zeros((self.env.num_agents, 2))
        try:
            # 等待action服务器上线
            if not self.control_client.wait_for_server(timeout_sec=0.1):
                self.get_logger().warn('Action server not found, use default control [0,0]')
                return default_action
            # 构造空Goal从而仅触发请求，无参数
            goal_msg = AgentControl.Goal()
            future = self.control_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn('Action goal rejected by server, use default control [0,0]')
                return default_action
            # 获取结果
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result
            if result.success:
                ad_action_list = [[single_action.ax, single_action.delta] for single_action in result.ad_action]
                ad_action = jnp.array(ad_action_list, dtype=jnp.float32)
                assert ad_action.shape == (self.env.num_agents, result.action_dim), \
                    self.get_logger().warn(f'Control shape mismatch: {ad_action.shape} vs ({self.env.num_agents},{result.action_dim})')
                self.get_logger().debug(
                    f'Success get multi-agent control (shape: {ad_action.shape}): {ad_action}')
                return ad_action
            else:
                self.get_logger().warn(
                    f'Action result invalid (success={result.success}, dim={result.action_dim}), use default [0,0]')
                return default_action
        except Exception as e:
            self.get_logger().warn(f'Failed to get control cmd: {e}, use default [0,0]')
            return default_action

    def publish_state(self, graph: MVEEnvGraphsTuple, reward: float, cost: float):
        """发布自车+障碍车状态到/ros_env"""
        aS_agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=self.env.num_agents)
        aS_goal_states = graph.type_states(type_idx=MVE.GOAL, n_type=self.env.num_agents)
        oS_obst_states = graph.type_states(type_idx=MVE.OBST, n_type=self.env.num_obsts)
        agent_states_np = np.asarray(aS_agent_states)
        goal_states_np = np.asarray(aS_goal_states)
        obst_states_np = np.asarray(oS_obst_states)
        msg = StateAndEval()

        def _array_to_single_state(state_arr: np.ndarray) -> SingleObjectState:
            """将单个物体的状态数组（长度8）转为SingleObjectState msg"""
            single_state = SingleObjectState()
            single_state.x = state_arr[0]
            single_state.y = state_arr[1]
            single_state.vx = state_arr[2]
            single_state.vy = state_arr[3]
            single_state.theta = state_arr[4]
            single_state.dthetadt = state_arr[5]
            single_state.bw = state_arr[6]
            single_state.bh = state_arr[7]
            return single_state

        msg.aS_agent_states = [_array_to_single_state(state) for state in agent_states_np]
        msg.aS_goal_states = [_array_to_single_state(state) for state in goal_states_np]
        msg.oS_obst_states = [_array_to_single_state(state) for state in obst_states_np]
        msg.state_dim = 8

        msg.reward = reward
        msg.cost = cost

        self.state_pub.publish(msg)


    def terminate_sim(self):
        """终止仿真，发布终止信号（全0状态）"""
        self.is_running = False
        self.timer.cancel()
        # 发布终止信号（供action节点感知）
        terminate_state = StateAndEval()
        self.state_pub.publish(terminate_state)
        self.get_logger().info(f'Simulation finished! Total steps: {self.current_step}')
        rclpy.shutdown()


def main():
    rclpy.init()
    env_node = EnvNode()
    executor = SingleThreadedExecutor()
    executor.add_node(env_node)
    try:
        executor.spin()
    finally:
        env_node.destroy_node()
        executor.shutdown()


if __name__ == '__main__':
    main()