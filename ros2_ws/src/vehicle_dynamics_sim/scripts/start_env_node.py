import os
import datetime
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
import time
import yaml
import sys

from defmarl.env import make_env
from your_dynamics_model import VehicleDynamics  # 你的动力学模型
from your_reward_cost import calc_reward_cost  # 你的reward/cost计算
from ros2_ws.src.vehicle_dynamics_sim.msg import AgentControl, StateAndControlAndEval


class EnvNode(Node):
    def __init__(self, args):
        super().__init__ ('start_,env_node')

        if args.visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices

        import jax
        import jax.numpy as jnp
        import jax.random as jr

        from defmarl.algo import make_algo
        from defmarl.env import make_env
        from defmarl.trainer.data import Rollout
        from defmarl.trainer.utils import eval_rollout
        from defmarl.utils.utils import jax_jit_np, jax_vmap, parse_jax_array

        n_gpu = jax.local_device_count()
        print(f"> initializing EnvNode {args}")
        print(f"> Using {n_gpu} devices")

        stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

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
            num_obs=config.obs if args.obs is None else args.obs,
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
        self.state_pub = self.create_publisher(StateAndControlAndEval, '/ros_env', 10)
        # 创建Action客户端（向ros_action节点请求控制量）
        self.control_client = ActionClient(self, AgentControl, '/ros_action')
        # 定时器（40Hz执行step）
        self.timer = self.create_timer(self.env.dt, self.step_callback)

        # 6. 运行标记
        self.is_running = True
        self.get_logger().info('Env node initialized, max step: %d, freq: %dHz' % (self.env.max_step, 1/self.env.dt))


    def step_callback(self):
        """核心step逻辑：40Hz执行"""
        if not self.is_running or self.current_step >= self.env.max_step:
            self.terminate_sim()
            return

        step_start = time.time()
        # 获取控制量
        action = self.get_control_cmd()  # [acc, delta]
        # 执行动力学step
        next_graph, reward, cost, done, info = self.env.step(self.current_graph, action)
        self.current_graph = next_graph


        # 发布状态量
        self.publish_state(self.current_graph)

        # 6. 同步现实时间：补足时延
        step_elapsed = time.time() - step_start
        if step_elapsed < self.step_dt:
            time.sleep(self.step_dt - step_elapsed)

        # 7. 更新步数
        self.current_step += 1
        if self.current_step % 10 == 0:  # 每10步打印日志
            self.get_logger().debug(f'Step {self.current_step}/{self.max_step}, reward: {reward:.2f}, cost: {cost:.2f}')

    def get_control_cmd(self):
        """向action节点请求控制量，默认[0,0]"""
        try:
            # 等待action服务器上线
            if not self.control_client.wait_for_server(timeout_sec=1.0):
                self.get_logger().warn('Action server not found, use default control [0,0]')
                return [0.0, 0.0]
            # 发送控制量请求
            goal_msg = ControlCmd.Goal()
            future = self.control_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            if not goal_handle.accepted:
                return [0.0, 0.0]
            # 获取结果
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result
            if result.success:
                return [result.acc, result.delta]
        except Exception as e:
            self.get_logger().warn(f'Failed to get control cmd: {e}, use default [0,0]')
        return [0.0, 0.0]

    def publish_state(self, state):
        """发布自车+障碍车状态到/ros_env/vehicle_state"""
        msg = VehicleState()
        msg.x = state['x']
        msg.y = state['y']
        msg.v = state['v']
        msg.yaw = state['yaw']
        msg.obs_x = state['obs_x']
        msg.obs_y = state['obs_y']
        msg.obs_v = state['obs_v']
        msg.obs_yaw = state['obs_yaw']
        self.state_pub.publish(msg)

    def publish_eval(self, reward, cost):
        """发布评估量到/ros_eval/metrics"""
        msg = EvalMetrics()
        msg.reward = reward
        msg.cost = cost
        msg.step = self.current_step
        self.eval_pub.publish(msg)

    def terminate_sim(self):
        """终止仿真，发布终止信号（全0状态）"""
        self.is_running = False
        self.timer.cancel()
        # 发布终止信号（供action节点感知）
        terminate_state = VehicleState()
        self.state_pub.publish(terminate_state)
        self.get_logger().info(f'Simulation finished! Total steps: {self.current_step}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
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