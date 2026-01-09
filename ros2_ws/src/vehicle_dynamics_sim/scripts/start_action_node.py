import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 由于是测试，没有进行数据并行的必要，强制使用第0个GPU
import rclpy
import yaml
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import Optional
from rclpy.node import Node
from rclpy.action import ActionServer
from defmarl.algo import make_algo
from defmarl.env import make_env
from defmarl.env.mve import MVEEnvState
from defmarl.utils.utils import parse_jax_array
from vehicle_dynamics_sim.msg import StateAndEval
from vehicle_dynamics_sim.action import AgentControl, SingleAgentControl


class ActionNode(Node):
    def __init__(self, args):
        super().__init__('start_action_node')

        # 声明所有参数（整合env参数+自身独有参数)
        # 公共必填参数
        self.declare_parameter('path', '')
        # 公共可选参数（与env节点一致）
        self.declare_parameter('num_agents', None)
        self.declare_parameter('env', None)
        self.declare_parameter('full_observation', False)
        self.declare_parameter('cpu', False)
        self.declare_parameter('max_step', None)
        self.declare_parameter('seed', 1234)
        self.declare_parameter('debug', False)
        self.declare_parameter('area_size', '')
        # 自身独有可选参数
        self.declare_parameter('from_iter', None)
        self.declare_parameter('stochastic', False)

        # 获取所有参数（封装为对象）
        self.args = self.get_all_parameters()

        # 校验必填参数
        if not self.args.path:
            self.get_logger().fatal('Parameter "path" is required! Please set it in launch file.')
            rclpy.shutdown()
            raise SystemExit('Missing required parameter: path')


        n_gpu = jax.local_device_count()
        print(f"> initializing ActionNode {args}")
        print(f"> Using {n_gpu} devices")

        # set up environment variables and seed
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if args.cpu:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
        if args.debug:
            jax.config.update("jax_disable_jit", True)
        np.random.seed(args.seed)
        self.key = jr.PRNGKey(args.seed)

        # load config
        if args.path is not None:
            with open(os.path.join(args.path, "config.yaml"), "r") as f:
                config = yaml.load(f, Loader=yaml.UnsafeLoader)

        # 加载神经网络模型
        path = args.path
        model_path = os.path.join(path, "models")
        if args.from_iter is None:
            models = os.listdir(model_path)
            from_iter = max([int(model) for model in models if model.isdigit()])
        else:
            from_iter = args.from_iter
        print("from_iter: ", from_iter)

        # create environments
        num_agents = config.num_agents if args.num_agents is None else args.num_agents
        env = make_env(
            env_id=config.env if args.env is None else args.env,
            num_agents=num_agents,
            num_obs=config.obs,
            max_step=args.max_step,
            full_observation=args.full_observation,
            area_size=config.area_size if args.area_size is None else args.area_size,
        ) # 和EnvNode一样的环境，相当于算法自己的环境备份
        self.env = env

        algo = make_algo(
            algo=config.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            cost_weight=config.cost_weight,
            actor_gnn_layers=config.gnn_layers,
            critic_gnn_layers=config.gnn_layers,
            Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
            lr_actor=config.lr_actor,
            lr_cbf=config.lr_critic,
            max_grad_norm=2.0,
            seed=config.seed,
            use_rnn=config.use_rnn,
            rnn_layers=config.rnn_layers,
            use_lstm=config.use_lstm,
        )
        algo.load(model_path, from_iter)
        self.algo = algo
        self.stochastic = args.stochastic

        if args.stochastic:
            def act_fn(x, z, rnn_state, key):
                action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
                return action, new_rnn_state
            act_fn = act_fn
        else:
            act_fn = algo.act
        self.act_fn = jax.jit(act_fn)

        z_fn = algo.get_opt_z if hasattr(algo, "get_opt_z") else None
        self.z_fn = z_fn

        init_rnn_state = algo.init_rnn_state
        if hasattr(algo, "init_Vh_rnn_state"):
            init_Vh_rnn_state = algo.init_Vh_rnn_state
        else:
            init_Vh_rnn_state = None
        self.actor_rnn_state = init_rnn_state
        self.Vh_rnn_state = init_Vh_rnn_state

        self.default_control = jnp.zeros((self.algo.n_agents, self.algo.action_dim))


        # 订阅状态量
        self.state_sub = self.create_subscription(StateAndEval, '/ros_env', self.state_callback,0)

        # 创建Action服务器（响应控制量请求）
        self.control_server = ActionServer(self, AgentControl, '/ros_action', self.control_callback)

        # 缓存最新状态
        self.latest_graph = None
        # 运行标记
        self.is_running = True
        self.get_logger().info('Action node initialized, loaded model from: %s' % model_path)

    def get_all_parameters(self):
        """获取所有ROS2参数，封装为对象"""
        class ActionArgs:
            path:str
            num_agents:Optional[int]
            env:Optional[int]
            full_observation:bool
            cpu:bool
            max_step:Optional[int]
            debug:bool
            area_size:jnp.ndarray
            from_iter:Optional[int]
            stochastic:bool
        args = ActionArgs()
        # 赋值公共参数
        args.path = self.get_parameter('path').get_parameter_value().string_value
        args.num_agents = self.get_parameter('num_agents').get_parameter_value().integer_value
        args.env = self.get_parameter('env').get_parameter_value().string_value
        args.full_observation = self.get_parameter('full_observation').get_parameter_value().bool_value
        args.cpu = self.get_parameter('cpu').get_parameter_value().bool_value
        args.max_step = self.get_parameter('max_step').get_parameter_value().integer_value
        args.seed = self.get_parameter('seed').get_parameter_value().integer_value
        args.debug = self.get_parameter('debug').get_parameter_value().bool_value
        area_size_str = self.get_parameter('area_size').get_parameter_value().string_value
        args.area_size = parse_jax_array(area_size_str) if area_size_str else None
        # 赋值自身独有参数
        args.from_iter = self.get_parameter('from_iter').get_parameter_value().integer_value
        args.stochastic = self.get_parameter('stochastic').get_parameter_value().bool_value
        # 修正None值
        for param in ['num_agents', 'env', 'max_step', 'from_iter']:
            val = getattr(args, param)
            if val == 0 or val == '':
                setattr(args, param, None)
        return args


    def state_callback(self, msg:StateAndEval):
        """订阅状态量，缓存最新状态；检测终止信号"""
        # 缓存状态
        aS_agent_states_list = [[single_state.x, single_state.y, single_state.vx, single_state.vy, single_state.theta, \
            single_state.dthetadt, single_state.bw, single_state.bh] for single_state in msg.aS_agent_states]
        aS_goal_states_list = [[single_state.x, single_state.y, single_state.vx, single_state.vy, single_state.theta, \
            single_state.dthetadt, single_state.bw, single_state.bh] for single_state in msg.aS_goal_states]
        oS_obst_states_list = [[single_state.x, single_state.y, single_state.vx, single_state.vy, single_state.theta, \
            single_state.dthetadt, single_state.bw, single_state.bh] for single_state in msg.oS_obst_states]
        aS_agent_states = jnp.array(aS_agent_states_list, dtype=jnp.float32)
        aS_goal_states = jnp.array(aS_goal_states_list, dtype=jnp.float32)
        oS_obst_states = jnp.array(oS_obst_states_list, dtype=jnp.float32)

        env_state = MVEEnvState(aS_agent_states, aS_goal_states, oS_obst_states)
        self.latest_graph = self.env.get_graph(env_state)

        # 检测终止信号（状态全0）
        if all(aS_agent_states == 0) and all(aS_goal_states == 0) and all(oS_obst_states == 0):
            self.is_running = False
            self.get_logger().info('Received terminate signal, stopping action node...')
            rclpy.shutdown()


    def control_callback(self, goal_handle):
        """Action服务器回调：输入状态→神经网络输出控制量"""
        if not self.is_running:
            result = AgentControl.Result()
            result.success = False
            result.message = 'Action node stopped'
            goal_handle.abort()
            return result

        # 无状态则返回默认控制量
        if self.latest_graph is None:
            self.get_logger().warn('No state received, use default control [0,0]')
            action = self.default_control
        else:
            if self.z_fn is not None:
                z, Vh_rnn_state = self.z_fn(self.latest_graph, self.Vh_rnn_state)
                z_max = np.max(z, axis=0)
                z = jnp.repeat(z_max[None], self.algo.n_agents, axis=0)
                self.Vh_rnn_state = Vh_rnn_state
            else:
                z = -(self.env.reward_max+self.env.reward_min)/2 * jnp.ones((self.algo.n_agents, 1))
                # 对于informarl和informarl-lagr，直接在每次测试时使用固定的z即可
            if not self.stochastic:
                action, actor_rnn_state = self.act_fn(self.latest_graph, z, self.actor_rnn_state)
            else:
                action, actor_rnn_state = self.act_fn(self.latest_graph, z, self.actor_rnn_state, self.key)

            # 更新rnn_state
            self.actor_rnn_state = actor_rnn_state

        # 构造响应
        result = AgentControl.Result()

        def _array_to_single_action(a_action: np.ndarray) -> SingleAgentControl:
            """将单个物体的动作数组（长度8）转为SingleAgentControl action"""
            single_action = SingleAgentControl()
            single_action.ax = a_action[0]
            single_action.delta = a_action[1]
            return single_action

        ad_action_np = np.asarray(action)
        result.ad_action = [_array_to_single_action(a_action_np) for a_action_np in ad_action_np]
        result.success = True
        result.message = f'Control: acc={ad_action_np[:,0]}, delta={ad_action_np[:,1]}'
        goal_handle.succeed()

        return result


def main():
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument("--path", type=str, required=True)
    # optional arguments
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--from-iter", type=int, default=None)
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("-z", type=str, default=None)
    parser.add_argument("--area-size", type=parse_jax_array, default=None,
                        help='输入jax数组，一维用逗号分隔（如10,20），二维用分号+逗号（如10,20;30,40）')
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-devices", type=str, default=None)
    args = parser.parse_args()

    rclpy.init()
    action_node = ActionNode(args)
    rclpy.spin(action_node)
    action_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()