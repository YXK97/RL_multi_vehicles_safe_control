import pathlib
import jax
import jax.random as jr
import jax.numpy as jnp
import functools as ft
import numpy as np

from typing import Optional, Tuple
from typing_extensions import override
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Reward, Cost, Array, State, Path
from ..utils.utils import tree_index, MutablePatchCollection, save_anim
from .mve import MVE, MVEEnvState, MVEEnvGraphsTuple

INF = jnp.inf

class MVELaneChangeAndOverTake(MVE):
    """该任务使用agent位姿和预设轨迹的偏移量、加减速度和方向盘转角的大小作为的reward的度量，
    scaling factor作为cost的度量，每个agent分配一个goal并规划出一条轨迹（五次多项式），
    环境为两车道，障碍车均沿车道作匀速直线运动"""

    PARAMS = {
        "ego_lf": 0.905, # m
        "ego_lr": 1.305, # m
        "ego_bb_size": jnp.array([2.21, 1.48]), # bounding box的[width, height] m
        "comm_radius": 30,
        "obst_bb_size": jnp.array([4.18, 1.99]), # bounding box的[width, height] m
        "collide_extra_bias": 0.1, # 用于计算cost时避碰的margin m

        # [x_l, x_h, y_l, y_h, vx_l, vx_h, vy_l, vy_h, theta_l, theta_h, delta_l, delta_h, bbw_l, bbw_h, bbh_l, bbh_h, \
        # a0_l, a0_h, a1_l, a1_h, a2_l, a2_h, a3_l, a3_h, a4_l, a4_h, a5_l, a5_h]
        # 单位：x,y,bbw,bbh: m  vx,vy: km/h,  theta,delta: degree,  其它: 无
        # 速度v的限制需要计算sqrt(vx^2+vy^2)， 速率范围为[0, 100]
        "default_state_range": jnp.array([-100., 100., -3., 3., -100., 100., -100., 100., 0., 360., -10., 10., -INF, INF,
        -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF]), # 默认范围，用于指示正常工作的状态范围
        "rollout_state_range": jnp.array([-120., 120., -10., 10., -100., 100., -100., 100., 0., 360., -10., 10., -INF, INF,
        -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF]), # rollout过程中的限制，强制约束
        "agent_init_state_range": jnp.array([-100., 50., -1.5, 1.5, 0., 80., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        # 用于agent初始化的状态范围，其中y坐标为离散约束，agent初始化的y坐标只能位于-1.5或1.5
        "goal_state_range": jnp.array([100., 100., -1.5, 1.5, 60., 100., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        # 随机生成goal时的状态范围，其中y坐标为离散约束，goal的y坐标只能位于-1.5或1.5，xy和theta仅用于初始化轨迹参数，轨迹初始化后goal除了
        # vx其余状态均置0，不参与通信计算，vx仅作为速率目标参与通信计算
        "obst_state_range": jnp.array([-100., 80., -1.5, 1.5, 10., 120., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        # 随机生成obst时的状态范围，其中y坐标为离散约束，obst的y坐标只能位于-1.5或1.5，obst沿x轴正向以vx作匀速直线运动
        # 以上agent、goal、obst的状态约束参数均只有前6项有效

        "n_obsts": 2, # 本环境使用两个obst，每根车道一个

        "dist2path_bias": 0.05, # 用于判断agent是否沿轨迹行驶 m
        "theta2path_bias": 0.995, # 用于判断agent航向角是否满足轨迹的要求，即agent方向向量和轨迹方向向量夹角的cos是否大于0.995（是否小于5度）
        "delta2mid_bias": 5 # 用于判断前轮转角是否是中性，小于±5°就不惩罚
    }
    PARAMS.update({
        "ego_radius": jnp.linalg.norm(PARAMS["ego_bb_size"]/2), # m
        "ego_L": PARAMS["ego_lf"]+PARAMS["ego_lr"] # m
    })
    if "obst_bb_size" in PARAMS and PARAMS["obst_bb_size"].shape == (2,):
        PARAMS.update({"obst_radius": jnp.linalg.norm(PARAMS["obst_bb_size"]/2)})

    def __init__(self,
                 num_agents: int,
                 area_size: Optional[float] = None,
                 max_step: int = 256,
                 max_travel: Optional[float] = None,
                 dt: float = 0.05,
                 params: dict = None
                 ):
        params = MVELaneChangeAndOverTake.PARAMS if params is None else params
        super(MVELaneChangeAndOverTake, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        assert self.params["n_obsts"] == MVELaneChangeAndOverTake.PARAMS["n_obsts"], "本环境只接受2个障碍物的设置！"

    @override
    @property
    def state_dim(self) -> int:
        return 14 # x y vx vy theta delta bb_w bb_h a0 a1 a2 a3 a4 a5

    @override
    @property
    def node_dim(self) -> int:
        return 17  # state_dim(14)  indicator(3): agent: 001, goal: 010, obstacle: 100, pad: 00-1

    @override
    @property
    def edge_dim(self) -> int:
        return 8 # state_diff[:7]: x_diff, y_diff, vx_diff, vy_diff, theta_diff, delta_diff, bb_w_diff, bb_h_diff

    @override
    @property
    def action_dim(self) -> int:
        return 2  # a：车辆纵向加速度（m/s^2） omega：前轮转角角速度（逆时针为正，degree/s）

    @property
    def reward_min(self) -> float:
        return -(jnp.linalg.norm(jnp.array([self.area_size[jnp.array([0, 2])] - self.area_size[jnp.array([1, 3])]])) * 0.01) * self.max_episode_steps

    @override
    def reset(self, key: Array) -> GraphsTuple:
        """先生成obstacle，将obstacle视为agent，通过cost计算是否valid
        再生成agent和goal，将之前生成的obstacle还原为obstacle，利用cost计算是否valid
        最后使用五次多项式拟合初始路径"""
        x_l_idx = 0; x_h_idx = 1; y_l_idx = 2; y_h_idx = 3; vx_l_idx = 4; vx_h_idx = 5

        if self.params["n_obsts"] > 0:
            # randomly generate obstacles
            def get_obst(inp):
                this_key, state_range, _ = inp
                use_key1, use_key2, this_key = jr.split(this_key, 3)
                o_xvx = jr.uniform(use_key1, shape=(self.params["n_obsts"], 2),
                            minval=jnp.stack([obst_state_range[x_l_idx], obst_state_range[vx_l_idx]], axis=0),
                            maxval=jnp.stack([obst_state_range[x_h_idx], obst_state_range[vx_h_idx]], axis=0))
                o_y = jr.choice(use_key2, obst_state_range[jnp.array([y_l_idx, y_h_idx])],
                            shape=(self.params["n_obsts"], 1))
                o_other_0 = jnp.zeros((self.params["n_obsts"], self.state_dim-3), dtype=jnp.float32)
                o_state = jnp.concatenate([jnp.insert(o_xvx, 1, o_y, axis=1), o_other_0], axis=1)
                return this_key, state_range, o_state

            def non_valid_obst(inp):
                "根据cost判断是否valid"
                _, _, this_candidates = inp
                empty_obsts = jnp.empty((0, self.state_dim))
                tmp_state = MVEEnvState(this_candidates, this_candidates, empty_obsts)
                tmp_graph = self.get_graph(tmp_state, obst_as_agent=True)
                cost = self.get_cost(tmp_graph)
                return jnp.max(cost) > -0.5

            def get_valid_obsts(state_range, key):
                use_key1, use_key2, this_key = jr.split(key, 3)
                o_xvx_can = jr.uniform(use_key1, shape=(self.params["n_obsts"], 2),
                                   minval=jnp.stack([obst_state_range[x_l_idx], obst_state_range[vx_l_idx]], axis=0),
                                   maxval=jnp.stack([obst_state_range[x_h_idx], obst_state_range[vx_h_idx]], axis=0))
                o_y_can = jr.choice(use_key2, obst_state_range[jnp.array([y_l_idx, y_h_idx])],
                                shape=(self.params["n_obsts"], 1))
                o_other_0_can = jnp.zeros((self.params["n_obsts"], self.state_dim - 3), dtype=jnp.float32)
                o_state_can = jnp.concatenate([jnp.insert(o_xvx_can, 1, o_y_can, axis=1), o_other_0_can], axis=1)
                _, _, valid_obsts = jax.lax.while_loop(non_valid_obst, get_obst, (this_key, state_range, o_state_can))
                return valid_obsts

            if "obst_state_range" in self.params and self.params["obst_state_range"] is not None:
                obst_state_range = self.params["obst_state_range"]
            else:
                obst_state_range = self.params["default_state_range"]
            obst_key, key = jr.split(key, 2)
            obsts = get_valid_obsts(obst_state_range, obst_key)
        else:
            obsts = jnp.empty((0, self.state_dim))

        # randomly generate agents and goals
        def get_agent_goal(inp):
            this_key, state_range, _, obsts = inp
            use_key1, use_key2, this_key = jr.split(this_key, 3)
            a_xvx = jr.uniform(use_key1, shape=(self.num_agents, 2),
                               minval=jnp.stack([state_range[x_l_idx], state_range[vx_l_idx]], axis=0),
                               maxval=jnp.stack([state_range[x_h_idx], state_range[vx_h_idx]], axis=0))
            a_y = jr.choice(use_key2, state_range[jnp.array([y_l_idx, y_h_idx])],
                            shape=(self.num_agents, 1))
            a_other_0 = jnp.zeros((self.num_agents, self.state_dim - 3), dtype=jnp.float32)
            a_state = jnp.concatenate([jnp.insert(a_xvx, 1, a_y, axis=1), a_other_0], axis=1)
            return this_key, state_range, a_state, obsts

        def non_valid_agent_goal(inp):
            "根据cost判断是否valid"
            _, _, this_candidates, obsts = inp
            tmp_state = MVEEnvState(this_candidates, this_candidates, obsts)
            tmp_graph = self.get_graph(tmp_state)
            cost = self.get_cost(tmp_graph)
            return jnp.max(cost) > -0.5

        def get_valid_agent_goal(state_range, key, obsts):
            use_key1, use_key2, this_key = jr.split(key, 3)
            a_xvx_can = jr.uniform(use_key1, shape=(self.num_agents, 2),
                               minval=jnp.stack([state_range[x_l_idx], state_range[vx_l_idx]], axis=0),
                               maxval=jnp.stack([state_range[x_h_idx], state_range[vx_h_idx]], axis=0))
            a_y_can = jr.choice(use_key2, state_range[jnp.array([y_l_idx, y_h_idx])],
                            shape=(self.num_agents, 1))
            a_other_0_can = jnp.zeros((self.num_agents, self.state_dim - 3), dtype=jnp.float32)
            a_state_can = jnp.concatenate([jnp.insert(a_xvx_can, 1, a_y_can, axis=1), a_other_0_can], axis=1)
            _, _, valid_targets, _ = jax.lax.while_loop(non_valid_agent_goal, get_agent_goal,
                                    (this_key, state_range, a_state_can, obsts))
            return valid_targets

        if "goal_state_range" in self.params and self.params["goal_state_range"] is not None:
            goal_state_range = self.params["goal_state_range"]
        else:
            goal_state_range = self.params["default_state_range"]
        goal_key, key = jr.split(key, 2)
        goals = get_valid_agent_goal(goal_state_range, goal_key, obsts)

        if "agent_init_state_range" in self.params:
            if self.params["agent_init_state_range"] is not None:
                agent_init_state_range = self.params["agent_init_state_range"]
            else:
                agent_init_state_range = self.params["default_state_range"]
        else:
            agent_init_state_range = self.params["default_state_range"]
        agent_key = key
        agents = get_valid_agent_goal(agent_init_state_range, agent_key, obsts)

        env_state = MVEEnvState(agents, goals, obsts)

        return self.get_graph_init_path(env_state)

    def get_reward(self, graph: MVEEnvGraphsTuple, action: Action) -> Reward:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = graph.env_states.obstacle.shape[0]

        agents_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        goals_states = graph.type_states(type_idx=MVE.GOAL, n_type=num_goals)
        a_goal_v = jnp.linalg.norm(goals_states[:, 2:4], axis=1)
        a2_pos = agents_states[:, :2]
        a_path = agents_states[:, 8:]
        a_theta = agents_states[:, 4]
        a_delta = agents_states[:, 5]
        a_v = jnp.linalg.norm(agents_states[:, 2:4], axis=1)

        reward = jnp.zeros(()).astype(jnp.float32)
        # 循迹奖励： 位置+角度
        # 位置奖励
        a_x = a2_pos[:, 0]
        zeros = jnp.zeros_like(a_x)
        ones = jnp.ones_like(a_x)
        a_paths_y = (jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(
            a_path, jnp.stack([ones, a_x, a_x**2, a_x**3, a_x**4, a_x**5], axis=1)))
        a2_paths_pos = jnp.stack([a_x, a_paths_y], axis=1)
        a_dist2paths = jnp.linalg.norm(a2_paths_pos - a2_pos, axis=-1)
        reward -= (a_dist2paths.mean()) * 0.01
        reward -= jnp.where(a_dist2paths > self.params["dist2path_bias"], 1., 0.).mean() * 0.005
        # 角度奖励
        a_theta_grad = a_theta * jnp.pi / 180
        a2_vec = jnp.stack([jnp.cos(a_theta_grad), jnp.sin(a_theta_grad)], axis=1)
        a_paths_derivative = jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(
            a_path, jnp.stack([zeros, ones, 2*a_x, 3*a_x**2, 4*a_x**3, 5*a_x**4], axis=1))
        a2_paths_vec = jnp.stack([ones, a_paths_derivative], axis=1) # 只能处理轨迹中车辆应当沿x轴正向前进的情况
        a2_paths_vec = a2_paths_vec / jnp.linalg.norm(a2_paths_vec, axis=1)
        a_theta2paths = jnp.einsum('ij,ij->i', a2_vec, a2_paths_vec)
        reward += (a_theta2paths.mean() - 1) * 0.01
        reward -= jnp.where(a_theta2paths < self.params["theta2path_bias"], 1., 0.).mean() * 0.005

        # 速度跟踪奖励
        reward -= jnp.abs(a_v - a_goal_v).mean() * 0.01

        # 动作奖励
        reward -= (action[:, 0]**2).mean() * 0.01
        reward -= (action[:, 1]**2).mean() * 0.01

        # 转向角中性奖励
        reward -= (jax.nn.relu(jnp.abs(a_delta) - self.params["delta2mid_bias"])**2).mean() * 0.01

        return reward

    def get_cost(self, graph: MVEEnvGraphsTuple) -> Cost:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = graph.env_states.obstacle.shape[0]

        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        obstacle_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)

        agent_nodes = graph.type_nodes(type_idx=MVE.AGENT, n_type=num_agents)
        agent_radius = jnp.linalg.norm(agent_nodes[0, 4:6] / 2)
        if num_obsts > 0:
            obstacle_nodes = graph.type_nodes(type_idx=MVE.OBST, n_type=num_obsts)
            obst_radius = jnp.linalg.norm(obstacle_nodes[0, 4:6] / 2)

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = agent_radius * 2 + self.params["collide_extra_bias"] - min_dist

        # collision between agents and obstacles
        if num_obsts == 0:
            obst_cost = -jnp.ones(num_agents)
        else:
            obstacle_pos = obstacle_states[:, :2]
            dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(obstacle_pos, 0), axis=-1)
            min_dist = jnp.min(dist, axis=1)
            obst_cost: Array = agent_radius + obst_radius + self.params["collide_extra_bias"] - min_dist

        """
        # 对于agent是否超出边界的判断
        if "rollout_state_range" in self.params and self.params["rollout_state_range"] is not None:
            rollout_state_range = self.params["rollout_state_range"]
        else:
            rollout_state_range = self.params["default_state_range"]
        agent_bound_cost_xl = rollout_state_range[0] - agent_pos[:, 0]
        agent_bound_cost_xh = -(rollout_state_range[1] - agent_pos[:, 0])
        agent_bound_cost_yl = rollout_state_range[2] - agent_pos[:, 1]
        agent_bound_cost_yh = -(rollout_state_range[3] - agent_pos[:, 1])
        agent_bound_cost = jnp.concatenate([agent_bound_cost_xl[:, None], agent_bound_cost_xh[:, None],
                                            agent_bound_cost_yl[:, None], agent_bound_cost_yh[:, None]], axis=1)

        cost = jnp.concatenate([agent_cost[:, None], obst_cost[:, None], agent_bound_cost], axis=1)
        assert cost.shape == (num_agents, self.n_cost)
        """

        cost = jnp.concatenate([agent_cost[:, None], obst_cost[:, None]], axis=1)
        assert cost.shape == (num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0)

        return cost

    @override
    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: Optional[dict] = None,
            n_goals: Optional[int] = None,
            **kwargs
    ) -> None:
        n_goals = self.num_agents if n_goals is None else n_goals

        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(20,
                                (self.area_size[3]+3-(self.area_size[2]-3))*20/(self.area_size[1]+3-(self.area_size[0]-3))+4)
                               , dpi=100)
        ax.set_xlim(self.area_size[0]-3, self.area_size[1]+3)
        ax.set_ylim(self.area_size[2]-3, self.area_size[3]+3)
        ax.set(aspect="equal")
        plt.axis("on")
        if viz_opts is None:
            viz_opts = {}

        # 画y轴方向的限制，即车道边界限制
        ax.axhline(y=self.area_size[2], linewidth=2, color='k')
        ax.axhline(y=self.area_size[3], linewidth=2, color='k')

        # 画x轴方向的限制
        ax.axvline(x=self.area_size[0], linewidth=2, color='k')
        ax.axvline(x=self.area_size[1], linewidth=2, color='k')

        # plot the first frame
        T_graph = rollout.graph
        graph0 = tree_index(T_graph, 0)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obst_color = "#8a0000"
        edge_goal_color = goal_color

        # plot obstacles
        if self.params["n_obsts"] > 0:
            obsts_state_bbsize = graph0.type_nodes(type_idx=MVE.OBST, n_type=self.params["n_obsts"])[:, :6]  # [n_obsts, 6] x,y,theta,v,width,height
            obsts_pos = obsts_state_bbsize[:, :2]
            obsts_theta = obsts_state_bbsize[:, 2]
            obsts_bb_size = obsts_state_bbsize[:, 4:6]
            obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)
            plot_obsts_arrow = [plt.Arrow(x=obsts_pos[i,0], y=obsts_pos[i,1],
                                          dx=jnp.cos(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                          dy=jnp.sin(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                          width=1, color=obst_color, alpha=1.0) for i in range(len(obsts_theta))]
            plot_obsts_rec = [plt.Rectangle(xy=tuple(obsts_pos[i,:]-obsts_bb_size[i,:]/2),
                                            width=obsts_bb_size[i,0], height=obsts_bb_size[i,1],
                                            angle=obsts_theta[i], rotation_point='center',
                                            color=obst_color, linewidth=0.0, alpha=0.6) for i in range(len(obsts_theta))]
            plot_obsts_cir = [plt.Circle(xy=(obsts_pos[i,0], obsts_pos[i,1]), radius=self.params["obst_radius"],
                                         color=obst_color, linewidth=0.0, alpha=0.3) for i in range(len(obsts_theta))]
            col_obsts = MutablePatchCollection(plot_obsts_arrow+plot_obsts_rec+plot_obsts_cir, match_original=True, zorder=5)
            ax.add_collection(col_obsts)

        # plot goals
        goals_state_bbsize = graph0.type_nodes(type_idx=MVE.GOAL, n_type=n_goals)[:, :6]
        goals_pos = goals_state_bbsize[:, :2]
        goals_theta = goals_state_bbsize[:, 2]
        goals_bb_size = goals_state_bbsize[:, 4:6]
        goals_radius = jnp.linalg.norm(goals_bb_size, axis=1)
        plot_goals_arrow = [plt.Arrow(x=goals_pos[i,0], y=goals_pos[i,1],
                                      dx=jnp.cos(goals_theta[i]*jnp.pi/180)*goals_radius[i]/2,
                                      dy=jnp.sin(goals_theta[i]*jnp.pi/180)*goals_radius[i]/2,
                                      width=goals_radius[i]/jnp.mean(obsts_radius),
                                      alpha=1.0, color=goal_color) for i in range(n_goals)]
        plot_goals_rec = [plt.Rectangle(xy=tuple(goals_pos[i,:]-goals_bb_size[i,:]/2),
                                        width=goals_bb_size[i,0], height=goals_bb_size[i,1],
                                        angle=goals_theta[i], rotation_point='center',
                                        color=goal_color, linewidth=0.0, alpha=0.6) for i in range(n_goals)]
        plot_goals_cir = [plt.Circle(xy=(goals_pos[i,0], goals_pos[i,1]), radius=self.params["ego_radius"],
                                     color=goal_color, linewidth=0.0, alpha=0.3) for i in range(n_goals)]
        col_goals = MutablePatchCollection(plot_goals_arrow+plot_goals_rec+plot_goals_cir, match_original=True, zorder=6)
        ax.add_collection(col_goals)

        # plot agents
        agents_node = graph0.type_nodes(type_idx=MVE.AGENT, n_type=self.num_agents)
        agents_state_bbsize = agents_node[:, :6]
        agents_pos = agents_state_bbsize[:, :2]
        agents_theta = agents_state_bbsize[:, 2]
        agents_bb_size = agents_state_bbsize[:, 4:6]
        agents_radius = jnp.linalg.norm(agents_bb_size, axis=1)
        plot_agents_arrow = [plt.Arrow(x=agents_pos[i, 0], y=agents_pos[i, 1],
                                       dx=jnp.cos(agents_theta[i] * jnp.pi / 180) * agents_radius[i]/2,
                                       dy=jnp.sin(agents_theta[i] * jnp.pi / 180) * agents_radius[i]/2,
                                       width=agents_radius[i] / jnp.mean(obsts_radius),
                                       alpha=1.0, color=agent_color) for i in range(self.num_agents)]
        plot_agents_rec = [plt.Rectangle(xy=tuple(agents_pos[i,:]-agents_bb_size[i,:]/2),
                                         width=agents_bb_size[i,0], height=agents_bb_size[i,1],
                                         angle=agents_theta[i], rotation_point='center',
                                         color=agent_color, linewidth=0.0, alpha=0.6) for i in range(self.num_agents)]
        plot_agents_cir = [plt.Circle(xy=(agents_pos[i,0], agents_pos[i,1]), radius=self.params["ego_radius"],
                                      color=agent_color, linewidth=0.0, alpha=0.3) for i in range(self.num_agents)]
        col_agents = MutablePatchCollection(plot_agents_arrow+plot_agents_rec+plot_agents_cir, match_original=True, zorder=7)
        ax.add_collection(col_agents)

        # 画出agent的五次多项式path
        agents_path = agents_node[:, 6:12]
        a_xs = jax.vmap(lambda xl, xh: jnp.linspace(xl, xh, 100), in_axes=(0, 0))(agents_pos[:, 0], goals_pos[:, 0])
        ones = jnp.ones_like(a_xs)
        a_X = jnp.stack([ones, a_xs, a_xs**2, a_xs**3, a_xs**4, a_xs**5], axis=1)
        a_ys = jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(agents_path, a_X)
        path_lines = []
        for xs, ys in zip(a_xs, a_ys):
            path_lines.append(np.column_stack([xs, ys]))
        path_collection = LineCollection(path_lines, colors='k',  linewidths=1.5, linestyles='--', alpha = 1.0, zorder=8)
        ax.add_collection(path_collection)

        # plot edges
        all_pos = graph0.states[:, :2]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents + n_goals + self.params["n_obsts"], axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
        e_is_goal = (self.num_agents <= graph0.senders) & (graph0.senders < self.num_agents + n_goals)
        e_is_goal = e_is_goal[~is_pad]
        e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
        col_edges = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(col_edges)

        # texts
        text_font_opts = dict(
            size=16,
            color="k",
            family="sans-serif",
            weight="normal",
            transform=ax.transAxes,
        )
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
        if Ta_is_unsafe is not None:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
        z_text = ax.text(0.5, 1.04, "z: []", va="bottom", ha="center", **text_font_opts)

        # add agent labels
        label_font_opts = dict(
            size=20,
            color="k",
            family="sans-serif",
            weight="normal",
            ha="center",
            va="center",
            transform=ax.transData,
            clip_on=True,
            zorder=8,
        )
        agent_labels = [ax.text(float(agents_pos[ii, 0]), float(agents_pos[ii, 1]), f"{ii}", **label_font_opts)
                        for ii in range(self.num_agents)]

        if "Vh" in viz_opts:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

        # init function for animation
        def init_fn() -> list[plt.Artist]:
            return [col_obsts, col_goals, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos_t = graph.states[:-1, :2] # 最后一个node是padding，不要
            n_theta_t = graph.states[:-1, 2]
            n_bb_size_t = graph.nodes[:-1, 4:6]
            n_radius = jnp.linalg.norm(n_bb_size_t, axis=1)

            # update agents' positions and labels
            for ii in range(self.num_agents):
                plot_agents_arrow[ii].set_data(x=n_pos_t[ii, 0], y=n_pos_t[ii, 1],
                                               dx=jnp.cos(n_theta_t[ii]*jnp.pi/180)*n_radius[ii]/2,
                                               dy=jnp.sin(n_theta_t[ii]*jnp.pi/180)*n_radius[ii]/2)
                plot_agents_rec[ii].set_xy(xy=tuple(n_pos_t[ii, :]-n_bb_size_t[ii, :]/2))
                plot_agents_rec[ii].set_angle(angle=n_theta_t[ii])
                plot_agents_cir[ii].set_center(xy=tuple(n_pos_t[ii, :]))
                agent_labels[ii].set_position(n_pos_t[ii, :])
            # update goals' positions
            for ii in range(n_goals):
                plot_goals_arrow[ii].set_data(x=n_pos_t[self.num_agents+ii, 0], y=n_pos_t[self.num_agents+ii, 1],
                                              dx=jnp.cos(n_theta_t[self.num_agents+ii]*jnp.pi/180)*n_radius[self.num_agents+ii]/2,
                                              dy=jnp.sin(n_theta_t[self.num_agents+ii]*jnp.pi/180)*n_radius[self.num_agents+ii]/2)
                plot_goals_rec[ii].set_xy(xy=tuple(n_pos_t[self.num_agents+ii, :]-n_bb_size_t[self.num_agents+ii, :]/2))
                plot_goals_rec[ii].set_angle(angle=n_theta_t[self.num_agents+ii])
                plot_goals_cir[ii].set_center(xy=tuple(n_pos_t[self.num_agents+ii, :]))
            # update obstacles' positions
            if self.params["n_obsts"] > 0:
                for ii in range(self.params["n_obsts"]):
                    plot_obsts_arrow[ii].set_data(x=n_pos_t[self.num_agents+n_goals+ii, 0],
                                                  y=n_pos_t[self.num_agents+n_goals+ii, 1],
                                                  dx=jnp.cos(n_theta_t[self.num_agents+n_goals+ii]*jnp.pi/180)*n_radius[
                                                      self.num_agents+n_goals+ii]/2,
                                                  dy=jnp.sin(n_theta_t[self.num_agents+n_goals+ii]*jnp.pi/180)*n_radius[
                                                      self.num_agents+n_goals+ii]/2)
                    plot_obsts_rec[ii].set_xy(xy=tuple(n_pos_t[self.num_agents+n_goals+ii, :]-n_bb_size_t[self.num_agents+n_goals+ii, :]/2))
                    plot_obsts_rec[ii].set_angle(angle=n_theta_t[self.num_agents+n_goals+ii])
                    plot_obsts_cir[ii].set_center(xy=tuple(n_pos_t[self.num_agents+n_goals+ii, :]))

            # update edges
            e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
            is_pad_t = np.any(e_edge_index_t == self.num_agents + n_goals + self.params["n_obsts"], axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
            e_is_goal_t = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents + n_goals)
            e_is_goal_t = e_is_goal_t[~is_pad_t]
            e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
            e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
            col_edges.set_segments(e_lines_t)
            col_edges.set_colors(e_colors_t)

            # update cost and safe labels
            if kk < len(rollout.costs):
                all_costs = ""
                for i_cost in range(rollout.costs[kk].shape[1]):
                    all_costs += f"    {self.cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
                all_costs = all_costs[:-2]
                cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
            else:
                cost_text.set_text("")
            if kk < len(Ta_is_unsafe):
                a_is_unsafe = Ta_is_unsafe[kk]
                unsafe_idx = np.where(a_is_unsafe)[0]
                safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
            else:
                safe_text[0].set_text("Unsafe: {}")

            kk_text.set_text("kk={:04}".format(kk))

            # Update the z text.
            z_text.set_text(f"z: {rollout.zs[kk]}")

            if "Vh" in viz_opts:
                Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

            return [col_obsts, col_goals, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)

    def edge_blocks(self, state: MVEEnvState) -> list[EdgeBlock]:
        num_agents = state.agent.shape[0]
        num_goals = state.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = state.obstacle.shape[0]

        agent_pos = state.agent[:, :2]
        id_agent = jnp.arange(num_agents)

        # agent - agent connection
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self.params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self.params["comm_radius"])
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        agent_goal_edges = []
        for i_agent in range(num_agents):
            agent_state_i = state.agent[i_agent]
            goal_state_i = state.goal[i_agent]
            agent_goal_feats_i = agent_state_i - goal_state_i
            agent_goal_edges.append(EdgeBlock(agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
                                              jnp.array([i_agent]), jnp.array([i_agent + num_agents])))

        # agent - obstacle connection
        agent_obst_edges = []
        if num_obsts > 0:
            obs_pos = state.obstacle[:, :2]
            poss_diff = agent_pos[:, None, :] - obs_pos[None, :, :]
            dist = jnp.linalg.norm(poss_diff, axis=-1)
            agent_obs_mask = jnp.less(dist, self.params["comm_radius"])
            id_obs = jnp.arange(num_obsts) + num_agents * 2
            state_diff = state.agent[:, None, :] - state.obstacle[None, :, :]
            agent_obst_edges = [EdgeBlock(state_diff, agent_obs_mask, id_agent, id_obs)]

        return [agent_agent_edges] + agent_goal_edges + agent_obst_edges
        # return agent_goal_edges + agent_obst_edges

    def generate_path(self, env_state: MVEEnvState) -> Path:
        """根据起点和终点求解五次多项式并写入graph"""
        agent_states = env_state.agent
        goal_states = env_state.goal
        @ft.partial(jax.jit)
        def A_b_create_and_solve(agent_state, goal_state) -> Path:
            x0 = agent_state[0]
            x1 = goal_state[0]
            A = jnp.array([[1, x0, x0**2,   x0**3,    x0**4,    x0**5],
                           [0,  1,  2*x0, 3*x0**2,  4*x0**3,  5*x0**4],
                           [0,  0,     2,    6*x0, 12*x0**2, 20*x0**3],
                           [1, x1, x1**2,   x1**3,    x1**4,    x1**5],
                           [0,  1,  2*x1, 3*x1**2,  4*x1**3,  5*x1**4],
                           [0,  0,     2,    6*x1, 12*x1**2, 20*x1**3],])
            y0 = agent_state[1]
            y1 = goal_state[1]
            t0 = agent_state[2]*jnp.pi/180
            t1 = goal_state[2]*jnp.pi/180
            b = jnp.array([y0, jnp.tan(t0), 0, y1, jnp.tan(t1), 0])
            coeff = jnp.linalg.solve(A, b)
            return coeff
        coeffs = jax.vmap(A_b_create_and_solve, in_axes=(0, 0))(agent_states, goal_states)
        return coeffs

    def get_graph_init_path(self, env_state: MVEEnvState, obst_as_agent:bool = False) -> MVEEnvGraphsTuple:
        num_agents = env_state.agent.shape[0]
        num_goals = env_state.goal.shape[0]
        num_obsts = env_state.obstacle.shape[0] # TODO: 为0时报错，但理论上可以为0
        assert num_agents > 0 and num_goals > 0, "至少需要设定agent和goal!"
        assert num_agents == num_goals, "每一个agent对应一个goal"
        # node features
        # states
        node_feats = jnp.zeros((num_agents + num_goals + num_obsts, self.node_dim))
        node_feats = node_feats.at[:num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, :self.state_dim].set(env_state.goal)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, :self.state_dim].set(env_state.obstacle)

        # bounding box长宽
        if obst_as_agent:
            node_feats = node_feats.at[:num_agents + num_goals, 4:6].set(self.params["obst_bb_size"])
        else:
            node_feats = node_feats.at[:num_agents + num_goals, 4:6].set(self.params["ego_bb_size"])
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, 4:6].set(self.params["obst_bb_size"])

        # 对agent设置五次多项式路径规划
        paths = self.generate_path(env_state)
        node_feats = node_feats.at[:num_agents, 6:12].set(paths)

        # indicators
        node_feats = node_feats.at[:num_agents, 14].set(1.0)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, 13].set(1.0)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, 12].set(1.0)

        # node type
        node_type = -jnp.ones((num_agents + num_goals + num_obsts), dtype=jnp.int32)
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[num_agents: num_agents + num_goals].set(MVE.GOAL)
        if num_obsts > 0:
            node_type = node_type.at[num_agents + num_goals:].set(MVE.OBST)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = jnp.concatenate([env_state.agent, env_state.goal], axis=0)
        if num_obsts > 0:
            states = jnp.concatenate([states, env_state.obstacle], axis=0)
        return GetGraph(node_feats, node_type, edge_blocks, env_state, states).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = self.params["rollout_state_range"][
            jnp.array([0, 2, 4, 6])]  # + jnp.array([0,-3,0,0]) # y方向增加可行宽度（相当于增加护墙不让车跨越，让车学会不要超出道路限制）
        upper_lim = self.params["rollout_state_range"][jnp.array([1, 3, 5, 7])]  # + jnp.array([0,3,0,0])
        return lower_lim, upper_lim

    @override
    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.array([0., -30.])[None, :].repeat(self.num_agents, axis=0) # v(只能直行), delta
        upper_lim = jnp.array([30., 30.])[None, :].repeat(self.num_agents, axis=0)
        return lower_lim, upper_lim
