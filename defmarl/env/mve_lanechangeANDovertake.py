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

from .mve import MVE, MVEEnvState, MVEEnvGraphsTuple
from .utils import process_lane_centers, process_lane_marks
from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Reward, Cost, Array, State, Path, AgentState, ObstState, Done, Info
from ..utils.utils import tree_index, MutablePatchCollection, save_anim, calc_2d_rot_matrix
from ..utils.scaling import scaling_calc, scaling_calc_bound

INF = jnp.inf

class MVELaneChangeAndOverTake(MVE):
    """该任务使用agent位姿和预设轨迹的偏移量、加减速度和方向盘转角的大小作为的reward的度量，
    scaling factor作为cost的度量，每个agent分配一个goal并规划出一条轨迹（五次多项式），
    环境为四车道，障碍车均沿车道作匀速直线运动"""

    PARAMS = {
        # 别克1949参数
        "ego_lf": 1.488, # m
        "ego_lr": 1.712, # m
        "ego_bb_size": jnp.array([4.2, 1.7]), # bounding box的[width, height] m
        "ego_m": 2045., # kg
        "ego_Iz": 5428., # kg*m^2
        "ego_Cf": 77850., # N/rad
        "ego_Cr": 76510., # N/rad
        "comm_radius": 30,
        "obst_bb_size": jnp.array([4.18, 1.99]), # bounding box的[width, height] m

        # [x_l, x_h, y_l, y_h, vx_l, vx_h, vy_l, vy_h, θ_l, θ_h, dθdt_l, dθdt_h, \
        # bbw_l, bbw_h, bbh_l, bbh_h, a0_l, a0_h, a1_l, a1_h, a2_l, a2_h, a3_l, a3_h, a4_l, a4_h, a5_l, a5_h]
        # 单位：x,y,bbw,bbh: m  vx,vy: km/h,  θ: °, dθdt: °/s, 其它: 无
        # 速度约束通过车身坐标系对纵向速度约束来进行
        "default_state_range": jnp.array([-100., 100., -6., 6., -INF, INF, -INF, INF, 0., 360., -INF, INF,
        -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF]), # 默认范围，用于指示正常工作的状态范围
        "rollout_state_range": jnp.array([-120., 100., -100., 100., -INF, INF, -INF, INF, 0., 360., -INF, INF,
        -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF]), # rollout过程中的限制，强制约束
        "rollout_state_b_range": jnp.array([-INF, INF, -INF, INF, 20., 100., -INF, INF, -INF, INF, -INF, INF,
        -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF, -INF, INF]), # rollout过程中在车身坐标系下状态约束，主要对纵向速度有约束，动力学模型不允许倒车
        "agent_init_state_range": jnp.array([-120., -80., -6., 6., 30., 80., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        # 用于agent初始化的状态范围，其中y坐标为离散约束，agent初始化的y坐标只能位于-4.5、-1.5、1.5或4.5
        "goal_state_range": jnp.array([100., 150., -6., 6., 40., 90., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        # 随机生成goal时的状态范围，其中y坐标为离散约束，goal的y坐标只能位于-4.5、-1.5、1.5或4.5，xy和theta仅用于初始化轨迹参数，轨迹初始化后goal除了
        # vx其余状态均置0，不参与通信计算，vx仅作为速率目标参与通信计算
        "obst_state_range": jnp.array([-150., 50., -6., 6., 10., 120., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        # 随机生成obst时的状态范围，其中y坐标为离散约束，obst的y坐标只能位于-4.5、-1.5、1.5或4.5，obst沿x轴正向以vx作匀速直线运动
        # 以上agent、goal、obst初始化的状态约束参数均只有前6项有效

        "n_obsts": 2, # 本环境使用两个obst
        "lane_width": 3, # 车道宽度，m

        "dist2path_bias": 0.1, # 用于判断agent是否沿轨迹行驶 m
        "theta2path_bias": 0.995, # 用于判断agent航向角是否满足轨迹的要求，即agent方向向量和轨迹方向向量夹角的cos是否大于0.995（是否小于5度）
        "delta2mid_bias": 3 # 用于判断前轮转角是否是中性，±3°以内就不惩罚
    }
    PARAMS.update({
        "ego_radius": jnp.linalg.norm(PARAMS["ego_bb_size"]/2), # m
        "ego_L": PARAMS["ego_lf"]+PARAMS["ego_lr"], # m
        "lane_centers": process_lane_centers(PARAMS["default_state_range"][2:4], PARAMS["lane_width"]), # 车道中心线y坐标 m
    })
    if "obst_bb_size" in PARAMS and PARAMS["obst_bb_size"].shape == (2,):
        PARAMS.update({"obst_radius": jnp.linalg.norm(PARAMS["obst_bb_size"]/2)})

    def __init__(self,
                 num_agents: int,
                 area_size: Optional[float] = None,
                 max_step: int = 128,
                 max_travel: Optional[float] = None,
                 dt: float = 0.05,
                 params: dict = None
                 ):
        area_size = MVELaneChangeAndOverTake.PARAMS["rollout_state_range"][:4] if area_size is None else area_size
        params = MVELaneChangeAndOverTake.PARAMS if params is None else params
        super(MVELaneChangeAndOverTake, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        assert self.params["n_obsts"] == MVELaneChangeAndOverTake.PARAMS["n_obsts"], "本环境只接受2个障碍物的设置！"

    @override
    @property
    def state_dim(self) -> int:
        return 14 # x y vx vy θ dθ/dt bb_w bb_h a0 a1 a2 a3 a4 a5

    @override
    @property
    def node_dim(self) -> int:
        return 17  # state_dim(14)  indicator(3): agent: 001, goal: 010, obstacle: 100, pad: 00-1

    @override
    @property
    def edge_dim(self) -> int:
        return 14 # Δstate: Δx, Δy, Δvx, Δvy, Δθ, Δdθ/dt, Δbb_w, Δbb_h, Δai

    @override
    @property
    def action_dim(self) -> int:
        return 2  # a：车辆纵向加速度（m/s^2） δ：前轮转角（逆时针为正，°）

    @override
    @property
    def reward_max(self):
        # return 0.5
        return 500 # debug

    @property
    def reward_min(self) -> float:
        # return -10 # TODO: fine-tune
        return 300 # debug

    @override
    @property
    def n_cost(self) -> int:
        return 4 # agent间碰撞(1) + agent-obstacle碰撞(1) + agent超出y轴范围(高+低，2)

    @override
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "bound exceeds y low", "bound exceeds y high"

    @override
    def reset(self, key: Array) -> GraphsTuple:
        """先生成obstacle，将obstacle视为agent，通过cost计算是否valid
        再生成agent和goal，将之前生成的obstacle还原为obstacle，利用cost计算是否valid
        最后使用五次多项式拟合初始路径"""
        x_l_idx = 0; x_h_idx = 1; vx_l_idx = 4; vx_h_idx = 5
        c_ycs = self.params["lane_centers"]
        obst_key, goal_key, agent_key = jr.split(key, 3)

        if self.params["n_obsts"] > 0:
            # randomly generate obstacles
            def get_obst(inp):
                this_key, state_range, _ = inp
                use_key, use_key2, new_key = jr.split(this_key, 3)
                o_xvx = jr.uniform(use_key, shape=(self.params["n_obsts"], 2),
                            minval=jnp.stack([state_range[x_l_idx], state_range[vx_l_idx]], axis=0),
                            maxval=jnp.stack([state_range[x_h_idx], state_range[vx_h_idx]], axis=0))
                o_y = jr.choice(use_key2, c_ycs, shape=(self.params["n_obsts"], 1))
                o_other_0 = jnp.zeros((self.params["n_obsts"], self.state_dim-3), dtype=jnp.float32)
                o_xyvx = jnp.insert(o_xvx, jnp.array([1]), o_y, axis=1)
                o_state = jnp.concatenate([o_xyvx, o_other_0], axis=1)
                return new_key, state_range, o_state

            def non_valid_obst(inp):
                "根据cost判断是否valid"
                _, _, this_candidates = inp
                empty_obsts = jnp.empty((0, self.state_dim))
                tmp_state = MVEEnvState(this_candidates, this_candidates, empty_obsts)
                tmp_graph = self.get_graph(tmp_state, obst_as_agent=True)
                cost = self.get_cost(tmp_graph)
                candidates_non_valid = jnp.max(cost) > -0.1
                # jax.debug.print("obst_cost={cost}", cost=jnp.max(cost))
                return candidates_non_valid

            def get_valid_obsts(state_range, key):
                use_key, use_key2, this_key = jr.split(key, 3)
                o_xvx_can = jr.uniform(use_key, shape=(self.params["n_obsts"], 2),
                                   minval=jnp.stack([state_range[x_l_idx], state_range[vx_l_idx]], axis=0),
                                   maxval=jnp.stack([state_range[x_h_idx], state_range[vx_h_idx]], axis=0))
                o_y_can = jr.choice(use_key2, c_ycs, shape=(self.params["n_obsts"], 1))
                o_other_0_can = jnp.zeros((self.params["n_obsts"], self.state_dim - 3), dtype=jnp.float32)
                o_xyvx_can = jnp.insert(o_xvx_can, jnp.array([1]), o_y_can, axis=1)
                o_state_can = jnp.concatenate([o_xyvx_can, o_other_0_can], axis=1)
                _, _, valid_obsts = jax.lax.while_loop(non_valid_obst, get_obst, (this_key, state_range, o_state_can))
                return valid_obsts

            if "obst_state_range" in self.params and self.params["obst_state_range"] is not None:
                obst_state_range = self.params["obst_state_range"]
            else:
                obst_state_range = self.params["default_state_range"]
            obsts = get_valid_obsts(obst_state_range, obst_key)
        else:
            obsts = jnp.empty((0, self.state_dim))

        # randomly generate agents and goals
        def get_agent(inp):
            this_key, state_range, _, obsts = inp
            use_key1, use_key2, new_key = jr.split(this_key, 3)
            a_xvx = jr.uniform(use_key1, shape=(self.num_agents, 2),
                               minval=jnp.stack([state_range[x_l_idx], state_range[vx_l_idx]], axis=0),
                               maxval=jnp.stack([state_range[x_h_idx], state_range[vx_h_idx]], axis=0))
            a_y = jr.choice(use_key2, c_ycs, shape=(self.num_agents, 1))
            a_other_0 = jnp.zeros((self.num_agents, self.state_dim - 3), dtype=jnp.float32)
            a_state = jnp.concatenate([jnp.insert(a_xvx, jnp.array([1]), a_y, axis=1), a_other_0], axis=1)
            # jax.debug.print("agent_key={new_key}", new_key=new_key)
            return new_key, state_range, a_state, obsts

        def non_valid_agent(inp):
            "根据cost判断是否valid"
            _, _, this_candidates, obsts = inp
            tmp_state = MVEEnvState(this_candidates, this_candidates, obsts)
            tmp_graph = self.get_graph(tmp_state)
            cost = self.get_cost(tmp_graph)
            candidates_non_valid = jnp.max(cost) > -0.1
            # jax.debug.print("agent_cost={cost}", cost=jnp.max(cost))
            return candidates_non_valid

        def get_valid_agent(state_range, key, obsts):
            use_key1, use_key2, this_key = jr.split(key, 3)
            a_xvx_can = jr.uniform(use_key1, shape=(self.num_agents, 2),
                               minval=jnp.stack([state_range[x_l_idx], state_range[vx_l_idx]], axis=0),
                               maxval=jnp.stack([state_range[x_h_idx], state_range[vx_h_idx]], axis=0))
            a_y_can = jr.choice(use_key2, c_ycs, shape=(self.num_agents,1))
            a_other_0_can = jnp.zeros((self.num_agents, self.state_dim - 3), dtype=jnp.float32)
            a_xyvx_can = jnp.insert(a_xvx_can, jnp.array([1]), a_y_can, axis=1)
            a_state_can = jnp.concatenate([a_xyvx_can, a_other_0_can], axis=1)
            _, _, valid_targets, _ = jax.lax.while_loop(non_valid_agent, get_agent,
                                    (this_key, state_range, a_state_can, obsts))
            # jax.debug.print("agent_key={new_key}", new_key=this_key)
            return valid_targets

        def get_goal(state_range, key):
            use_key1, use_key2 = jr.split(key, 2)
            a_xvx_can = jr.uniform(use_key1, shape=(self.num_agents, 2),
                                   minval=jnp.stack([state_range[x_l_idx], state_range[vx_l_idx]], axis=0),
                                   maxval=jnp.stack([state_range[x_h_idx], state_range[vx_h_idx]], axis=0))
            a_y_can = jr.choice(use_key2, c_ycs, shape=(self.num_agents, 1))
            a_other_0_can = jnp.zeros((self.num_agents, self.state_dim - 3), dtype=jnp.float32)
            a_state = jnp.concatenate([jnp.insert(a_xvx_can, jnp.array([1]), a_y_can, axis=1), a_other_0_can], axis=1)

            return a_state

        if "goal_state_range" in self.params and self.params["goal_state_range"] is not None:
            goal_state_range = self.params["goal_state_range"]
        else:
            goal_state_range = self.params["default_state_range"]
        goals = get_goal(goal_state_range, goal_key)

        if "agent_init_state_range" in self.params:
            if self.params["agent_init_state_range"] is not None:
                agent_init_state_range = self.params["agent_init_state_range"]
            else:
                agent_init_state_range = self.params["default_state_range"]
        else:
            agent_init_state_range = self.params["default_state_range"]
        agents = get_valid_agent(agent_init_state_range, agent_key, obsts)

        env_state = MVEEnvState(agents, goals, obsts)

        return self.get_graph_init_path(env_state)

    @override
    def agent_step_euler(self, aS_agent_states: AgentState, ad_action: Action) -> AgentState:
        """对agent，使用3-DOF自行车动力学模型"""
        assert ad_action.shape == (self.num_agents, self.action_dim)
        assert aS_agent_states.shape == (self.num_agents, self.state_dim)
        convert_vec_s = jnp.array([1, 1, 3.6, 3.6, 180/jnp.pi, 180/jnp.pi]) # eg. km/h / convert_vec -> m/s
        convert_vec_a = jnp.array([1, 180/jnp.pi]) # m/s²不变，° / convert_vec_a -> rad

        # 参数提取
        as_S = aS_agent_states[:, :6] # x, y, vx, vy, θ, dθ/dt
        a_theta = as_S[:, 4]  # degree
        as_S_metric = as_S / convert_vec_s # km/h->m/s, degree->rad, degree/s->rad/s
        ad_action_metric = ad_action / convert_vec_a

        # 旋转矩阵计算与广义旋转矩阵构造
        a22_Q = jax.vmap(calc_2d_rot_matrix, in_axes=(0,))(a_theta)
        def construct_transform_matrix(a22_Q):
            """从 (a, 2, 2) 的旋转矩阵 Q 构造 (a, 6, 6) 的分块矩阵。"""
            a = a22_Q.shape[0]
            a66_barQ = jnp.zeros((a, 6, 6))
            a66_barQ = a66_barQ.at[:, :2, :2].set(a22_Q)
            a66_barQ = a66_barQ.at[:, 2:4, 2:4].set(a22_Q)
            two2_I = jnp.eye(2)
            a66_barQ = a66_barQ.at[:, 4:6, 4:6].set(jnp.tile(two2_I, (a, 1, 1)))
            return a66_barQ
        ass_barQ = construct_transform_matrix(a22_Q)

        # 状态量从世界坐标系向车身坐标系转换与参数提取
        as_S_b_metric = jnp.einsum('aij, ai -> aj', ass_barQ, as_S_metric)
        a_vx_b_metric = as_S_b_metric[:, 2] # m/s
        a_vy_b_metric = as_S_b_metric[:, 3] # m/s
        a_dthetadt_metric = as_S_b_metric[:, 5] # rad/s
        a_ones = jnp.ones((self.num_agents,), dtype=jnp.float32)
        m = self.params["ego_m"] # kg
        lf = self.params["ego_lf"] # m
        lr = self.params["ego_lr"] # m
        Iz = self.params["ego_Iz"] # kg*m^2
        Cf = 2*self.params["ego_Cf"] # N/rad，自行车模型需要将轮胎的侧偏刚度×2以代表轴刚度
        Cr = 2*self.params["ego_Cr"] # N/rad

        # 车辆3自由度control affine(小转向角近似)动力学模型 状态更新
        as_f = jnp.stack([a_vx_b_metric,
                          a_vy_b_metric,
                          a_vy_b_metric * a_dthetadt_metric,
                          -a_vx_b_metric * a_dthetadt_metric - (Cf+Cr)*a_vy_b_metric/(m*a_vx_b_metric) + \
                            (Cr*lr-Cf*lf)*a_dthetadt_metric/(m*a_vx_b_metric),
                          a_dthetadt_metric,
                          (Cr*lr-Cf*lf)*a_vy_b_metric/(Iz*a_vx_b_metric) - \
                            (Cf*(lf**2)+Cr*(lr**2))*a_dthetadt_metric/(Iz*a_vx_b_metric)], axis=1)
        asd_g = jnp.zeros((self.num_agents, 6, self.action_dim), dtype=jnp.float32)
        asd_g = asd_g.at[:, 2, 0].set(a_ones)
        asd_g = asd_g.at[:, 2, 1].set(Cf*(a_vy_b_metric+lr*a_dthetadt_metric)/(m*a_vx_b_metric))
        asd_g = asd_g.at[:, 3, 1].set(a_ones*Cf/m)
        asd_g = asd_g.at[:, 5, 1].set(a_ones*Cf*lf/Iz)
        as_dS_b_metric = (as_f + jnp.einsum('asd, ad -> as', asd_g, ad_action_metric)) * self.dt
        as_S_b_new_unclip_metric = as_S_b_metric + as_dS_b_metric
        as_S_b_new_unclip = as_S_b_new_unclip_metric * convert_vec_s # 公制单位转换为任务单位
        aS_S_b_new_unclip = aS_agent_states.at[:, :6].set(as_S_b_new_unclip)
        assert aS_S_b_new_unclip.shape == (self.num_agents, self.state_dim)
        aS_S_b_new = self.clip_state_b(aS_S_b_new_unclip)
        as_S_b_new = aS_S_b_new[:, :6]
        as_S_new_unclip = jnp.einsum('aij, aj -> ai', ass_barQ, as_S_b_new)
        as_S_new_unclip = as_S_new_unclip.at[:, 4].set(as_S_new_unclip[:, 4] % 360)  # θ限制在[0, 360]°
        aS_S_new_unclip = aS_agent_states.at[:, :6].set(as_S_new_unclip)
        assert aS_S_new_unclip.shape == (self.num_agents, self.state_dim)
        aS_S_new = self.clip_state(aS_S_new_unclip)

        return aS_S_new


    def obst_step_euler(self, o_obst_states: ObstState) -> ObstState:
        """障碍车作匀速直线运动"""
        num_obsts = o_obst_states.shape[0]
        assert o_obst_states.shape == (num_obsts, self.state_dim)

        # 匀速直线运动模型
        o_x = o_obst_states[:, 0]
        o_vx = o_obst_states[:, 2]
        o_obst_states_new = o_obst_states.at[:, 0].set(o_x + o_vx/3.6*self.dt)

        assert o_obst_states_new.shape == (num_obsts, self.state_dim)
        return o_obst_states_new

    @override
    def step(
            self, graph: MVEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MVEEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
        goals = graph.type_states(type_idx=MVE.GOAL, n_type=self.num_agents)

        if self.params["n_obsts"] > 0:
            obst_states = graph.type_states(type_idx=MVE.OBST, n_type=self.params["n_obsts"])
            next_obst_states = self.obst_step_euler(obst_states)
        else:
            next_obst_states = None

        # calculate next graph
        action = self.transform_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_env_state = MVEEnvState(next_agent_states, goals, next_obst_states)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        return self.get_graph(next_env_state), reward, cost, done, info

    def get_reward(self, graph: MVEEnvGraphsTuple, ad_action: Action) -> Reward:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals

        aS_agents_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        aS_goals_states = graph.type_states(type_idx=MVE.GOAL, n_type=num_goals)
        # state: x, y, vx, vy, θ, dθ/dt, bb_w, bb_h, a0 ... a5
        a_goal_v = aS_goals_states[:, 2]
        a2_pos = aS_agents_states[:, :2]
        a2_v = aS_agents_states[:, 2:4]
        a_path = aS_agents_states[:, -6:]
        a_theta = aS_agents_states[:, 4]
        a22_Q = jax.vmap(calc_2d_rot_matrix, in_axes=(0,))(a_theta)
        a_vx_b = jnp.einsum('aij, ai -> aj', a22_Q, a2_v)[:, 0]

        reward = jnp.zeros(()).astype(jnp.float32)
        # 循迹奖励： 位置+角度
        # 位置奖励，只关注y方向控制误差
        a_x = a2_pos[:, 0]
        a_y = a2_pos[:, 1]
        zeros = jnp.zeros_like(a_x)
        ones = jnp.ones_like(a_x)
        a_paths_y = (jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(
            a_path, jnp.stack([ones, a_x, a_x**2, a_x**3, a_x**4, a_x**5], axis=1)))
        a2_paths_pos = jnp.stack([a_x, a_paths_y], axis=1)
        a_dist2paths = (a_y - a_paths_y)**2
        reward -= (a_dist2paths.mean()) * 0.1
        # reward -= jnp.where(a_dist2paths > self.params["dist2path_bias"], 1., 0.).mean() * 0.005
        # 角度奖励
        a_theta_grad = a_theta * jnp.pi / 180
        a2_vec = jnp.stack([jnp.cos(a_theta_grad), jnp.sin(a_theta_grad)], axis=1)
        a_paths_derivative = jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(
            a_path, jnp.stack([zeros, ones, 2*a_x, 3*a_x**2, 4*a_x**3, 5*a_x**4], axis=1))
        a2_paths_vec = jnp.stack([ones, a_paths_derivative], axis=1) # 只能处理轨迹中车辆应当沿x轴正向前进的情况
        a2_paths_vec = a2_paths_vec / jnp.linalg.norm(a2_paths_vec, axis=1)[:, None]
        a_theta2paths = jnp.einsum('ij,ij->i', a2_vec, a2_paths_vec)
        reward += (a_theta2paths.mean() - 1) * 0.01

        # 速度跟踪惩罚
        reward -= ((a_vx_b - a_goal_v)**2).mean() * 0.0001

        # 动作惩罚
        reward -= (ad_action[:, 0]**2).mean() * 0.0005
        reward -= (ad_action[:, 1]**2).mean() * 0.001

        return reward

    def get_cost(self, graph: MVEEnvGraphsTuple) -> Cost:
        """使用射线法计算的scaling factor：α为cost的评判指标，1-α<0安全，>=0不安全"""
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]

        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        # agent之间的scaling factor
        """
        if num_agents == 1:
            a_agent_cost = -jnp.ones((1,), dtype=jnp.float32)
        else :
            # 生成所有非重复的agent对（i,j），满足i < j（仅上三角，避免重复）
            i_pairs, j_pairs = jnp.triu_indices(n=num_agents, k=1)  # k=1表示排除对角线（i=j）
            state_i_pairs = agent_states[i_pairs, :]
            state_j_pairs = agent_states[j_pairs, :]
            alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
            # 构造对称的α矩阵（对角线设为无穷大，排除自身）
            alpha_matrix = jnp.full((num_agents, num_agents), INF)  # 初始化矩阵，填充无穷大
            # 填充上三角（i<j）和下三角（j<i），利用对称性避免重复计算
            alpha_matrix = alpha_matrix.at[i_pairs, j_pairs].set(alpha_pairs)
            alpha_matrix = alpha_matrix.at[j_pairs, i_pairs].set(alpha_pairs)
            # 步骤4：每个agent对应的行取最大值（即与其他agent的最小α，α越小越不安全）
            a_agent_cost = jnp.max(1-alpha_matrix, axis=1)
        """
        a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug

        # agent 和 obst 之间的scaling factor
        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
        else:
            obstacle_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)
            i_grid, j_grid = jnp.meshgrid(jnp.arange(num_agents), jnp.arange(num_obsts), indexing='ij')
            i_pairs = i_grid.ravel()  # [num_agents*num_obsts]
            j_pairs = j_grid.ravel()  # [num_agents*num_obsts]
            state_i_pairs = agent_states[i_pairs, :]
            state_j_pairs = obstacle_states[j_pairs, :]
            alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
            alpha_matrix = alpha_pairs.reshape((num_agents, num_obsts))
            a_obst_cost = jnp.max(1-alpha_matrix, axis=1)
        # a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug

        # agent 和 bound 之间的scaling factor，只对y方向有约束
        state_range = self.params["default_state_range"]
        yl = state_range[2]
        A = jnp.array([[0., 1.]])
        b = jnp.array([yl])
        a_bound_yl_cost = 1 - jax.vmap(scaling_calc_bound, in_axes=(0, None, None))(agent_states, A, b)

        yh = state_range[3]
        A = jnp.array([[0., -1.]])
        b = jnp.array([-yh])
        a_bound_yh_cost = 1 - jax.vmap(scaling_calc_bound, in_axes=(0, None, None))(agent_states, A, b)

        # a_bound_yl_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug
        # a_bound_yh_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug

        cost = jnp.stack([a_agent_cost, a_obst_cost, a_bound_yl_cost, a_bound_yh_cost], axis=1)
        assert cost.shape == (num_agents, self.n_cost)

        """
        # debug
        def jax_print():
            if num_obsts > 0:
                obst_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)
                jax.debug.print("======================= \n "
                                    "agent_states={agent_states} \n "
                                    "obst_states={obst_states} \n"
                                    "cost={cost} \n"
                                    "==================== \n ",
                                    agent_states=agent_states,
                                    obst_states=obst_states,
                                    cost=cost,)
            else:
                jax.debug.print("======================= \n "
                                    "agent_states={agent_states} \n "
                                    "cost={cost} \n"
                                    "==================== \n ",
                                    agent_states=agent_states,
                                    cost=cost)
        """

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
        fig, ax = plt.subplots(1, 1, figsize=(30,
                                (self.area_size[3]+3-(self.area_size[2]-3))*20/(self.area_size[1]+3-(self.area_size[0]-3))+4)
                               , dpi=100)
        ax.set_xlim(self.area_size[0], self.area_size[1])
        ax.set_ylim(self.area_size[2]-3, self.area_size[3]+3)
        ax.set(aspect="equal")
        plt.axis("on")
        if viz_opts is None:
            viz_opts = {}

        # 画y轴方向的限制，即车道边界限制
        ax.axhline(y=self.area_size[2], linewidth=2, color='k')
        ax.axhline(y=self.area_size[3], linewidth=2, color='k')

        # 画车道线
        two_yms_bold, l_yms_scatter = process_lane_marks(self.params["default_state_range"][2:4], self.params["lane_width"])
        ax.axhline(y=two_yms_bold[0], linewidth=1.5, color='b')
        ax.axhline(y=two_yms_bold[1], linewidth=1.5, color='b')
        if l_yms_scatter is not None:
            for ym in l_yms_scatter:
                ax.axhline(y=ym, linewidth=1, color='b', linestyle='--')

        # plot the first frame
        T_graph = rollout.graph
        graph0 = tree_index(T_graph, 0)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obst_color = "#8a0000"
        edge_goal_color = goal_color

        # plot obstacles
        if self.params["n_obsts"] > 0:
            obsts_state = graph0.type_nodes(type_idx=MVE.OBST, n_type=self.params["n_obsts"])
            # state: x, y, vx, vy, θ, dθ/dt, δ, bb_w, bb_h, a0 ... a5
            obsts_pos = obsts_state[:, :2]
            obsts_theta = obsts_state[:, 4]
            obsts_bb_size = obsts_state[:, 6:8]
            obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)
            plot_obsts_arrow = [plt.Arrow(x=obsts_pos[i,0], y=obsts_pos[i,1],
                                          dx=jnp.cos(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                          dy=jnp.sin(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                          width=1, color=obst_color, alpha=1.0) for i in range(len(obsts_theta))]
            plot_obsts_rec = [plt.Rectangle(xy=tuple(obsts_pos[i,:]-obsts_bb_size[i,:]/2),
                                            width=obsts_bb_size[i,0], height=obsts_bb_size[i,1],
                                            angle=obsts_theta[i], rotation_point='center',
                                            color=obst_color, linewidth=0.0, alpha=0.6) for i in range(len(obsts_theta))]
            col_obsts = MutablePatchCollection(plot_obsts_arrow+plot_obsts_rec, match_original=True, zorder=5)
            ax.add_collection(col_obsts)

        # plot agents
        agents_state = graph0.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
        # state: x, y, vx, vy, θ, dθ/dt, δ, bb_w, bb_h, a0 ... a5
        agents_pos = agents_state[:, :2]
        agents_theta = agents_state[:, 4]
        agents_bb_size = agents_state[:, 6:8]
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
        col_agents = MutablePatchCollection(plot_agents_arrow+plot_agents_rec, match_original=True, zorder=6)
        ax.add_collection(col_agents)

        # 画出agent的五次多项式path
        # state: x, y, vx, vy, θ, dθ/dt, δ, bb_w, bb_h, a0 ... a5
        agents_path = agents_state[:, 8:]
        a_xs = jax.vmap(lambda xl, xh: jnp.linspace(xl, xh, 100), in_axes=(0, 0))(
            agents_pos[:, 0], 300 * jnp.ones_like(agents_pos[:, 0]))
        ones = jnp.ones_like(a_xs)
        a_X = jnp.stack([ones, a_xs, a_xs**2, a_xs**3, a_xs**4, a_xs**5], axis=1)
        a_ys = jax.vmap(lambda a, x: jnp.dot(a, x), in_axes=(0, 0))(agents_path, a_X)
        path_lines = []
        for xs, ys in zip(a_xs, a_ys):
            path_lines.append(np.column_stack([xs, ys]))
        path_collection = LineCollection(path_lines, colors='k',  linewidths=1.5, linestyles='--', alpha = 1.0, zorder=7)
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
            return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos_t = graph.states[:-1, :2] # 最后一个node是padding，不要
            n_theta_t = graph.states[:-1, 4]
            n_bb_size_t = graph.nodes[:-1, 6:8]
            n_radius = jnp.linalg.norm(n_bb_size_t, axis=1)

            # update agents' positions and labels
            for ii in range(self.num_agents):
                plot_agents_arrow[ii].set_data(x=n_pos_t[ii, 0], y=n_pos_t[ii, 1],
                                               dx=jnp.cos(n_theta_t[ii]*jnp.pi/180)*n_radius[ii]/2,
                                               dy=jnp.sin(n_theta_t[ii]*jnp.pi/180)*n_radius[ii]/2)
                plot_agents_rec[ii].set_xy(xy=tuple(n_pos_t[ii, :]-n_bb_size_t[ii, :]/2))
                plot_agents_rec[ii].set_angle(angle=n_theta_t[ii])
                agent_labels[ii].set_position(n_pos_t[ii, :])
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

            return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

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

        """
        # agent - agent connection
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self.params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self.params["comm_radius"])
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)
        """

        # agent - goal connection
        agent_goal_edges = []
        for i_agent in range(num_agents):
            agent_state_i = state.agent[i_agent]
            goal_state_i = state.goal[i_agent]
            agent_goal_feats_i = agent_state_i - goal_state_i
            agent_goal_edges.append(EdgeBlock(agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
                                              jnp.array([i_agent]), jnp.array([i_agent + num_agents])))

        """
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
        """

        return agent_goal_edges # 跟踪任务debug

    def generate_path(self, env_state: MVEEnvState) -> Path:
        """根据起点和终点求解五次多项式并写入graph"""
        agent_states = env_state.agent
        goal_states = env_state.goal
        # state: x y vx vy θ dθdt bb_w bb_h a0 a1 a2 a3 a4 a5
        @jax.jit
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
            t0 = agent_state[4]*jnp.pi/180
            t1 = goal_state[4]*jnp.pi/180
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
        # state: x y vx vy θ dθdt δ bb_w bb_h a0 a1 a2 a3 a4 a5
        if obst_as_agent:
            node_feats = node_feats.at[:num_agents, 6:8].set(self.params["obst_bb_size"])
        else:
            node_feats = node_feats.at[:num_agents, 6:8].set(self.params["ego_bb_size"])
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, 6:8].set(self.params["obst_bb_size"])

        # 对agent设置五次多项式路径规划
        paths = self.generate_path(env_state)
        node_feats = node_feats.at[:num_agents, -9:-3].set(paths)

        # indicators
        node_feats = node_feats.at[:num_agents, -1].set(1.0)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, -2].set(1.0)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, -3].set(1.0)

        # node type
        node_type = -jnp.ones((num_agents + num_goals + num_obsts), dtype=jnp.int32)
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[num_agents: num_agents + num_goals].set(MVE.GOAL)
        if num_obsts > 0:
            node_type = node_type.at[num_agents + num_goals:].set(MVE.OBST)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        # 去掉goal坐标的影响
        node_feats = node_feats.at[num_agents: num_agents + num_goals, :2].set(jnp.zeros((2,), dtype=jnp.float32))
        states = jnp.concatenate([node_feats[:num_agents, :-3], node_feats[num_agents: num_agents + num_goals, :-3]],
                                 axis=0)
        if num_obsts > 0:
            states = jnp.concatenate([states, node_feats[num_agents + num_goals:, :-3]], axis=0)
            new_env_state = MVEEnvState(node_feats[:num_agents, :-3],
                                    node_feats[num_agents: num_agents + num_goals, :-3],
                                    node_feats[num_agents + num_goals:, :-3])
        else:
            new_env_state = MVEEnvState(node_feats[:num_agents, :-3],
                                        node_feats[num_agents: num_agents + num_goals, :-3],
                                        jnp.empty((0, self.state_dim)))

        return GetGraph(node_feats, node_type, edge_blocks, new_env_state, states).to_padded()

    @override
    def get_graph(self, env_state: MVEEnvState, obst_as_agent:bool = False) -> MVEEnvGraphsTuple:
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
        # state: x y vx vy θ dθdt δ bb_w bb_h a0 a1 a2 a3 a4 a5
        if obst_as_agent:
            node_feats = node_feats.at[:num_agents, 6:8].set(self.params["obst_bb_size"])
        else:
            node_feats = node_feats.at[:num_agents, 6:8].set(self.params["ego_bb_size"])
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, 6:8].set(self.params["obst_bb_size"])

        # indicators
        node_feats = node_feats.at[:num_agents, -1].set(1.0)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, -2].set(1.0)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, -3].set(1.0)

        # 对agent设置五次多项式路径规划
        paths = env_state.agent[:, -6:]
        node_feats = node_feats.at[:num_agents, -9:-3].set(paths)

        # node type
        node_type = -jnp.ones((num_agents + num_goals + num_obsts), dtype=jnp.int32)
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[num_agents: num_agents + num_goals].set(MVE.GOAL)
        if num_obsts > 0:
            node_type = node_type.at[num_agents + num_goals:].set(MVE.OBST)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        # 去掉goal坐标的影响
        node_feats = node_feats.at[num_agents: num_agents + num_goals, :2].set(jnp.zeros((2,), dtype=jnp.float32))
        states = jnp.concatenate([node_feats[:num_agents, :-3], node_feats[num_agents: num_agents + num_goals, :-3]],
                                 axis=0)
        if num_obsts > 0:
            states = jnp.concatenate([states, node_feats[num_agents + num_goals:, :-3]], axis=0)
            new_env_state = MVEEnvState(node_feats[:num_agents, :-3],
                                        node_feats[num_agents: num_agents + num_goals, :-3],
                                        node_feats[num_agents + num_goals:, :-3])
        else:
            new_env_state = MVEEnvState(node_feats[:num_agents, :-3],
                                        node_feats[num_agents: num_agents + num_goals, :-3],
                                        jnp.empty((0, self.state_dim)))
        return GetGraph(node_feats, node_type, edge_blocks, new_env_state, states).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """世界坐标系下的状态约束"""
        lower_lim = self.params["rollout_state_range"][jnp.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])]
        upper_lim = self.params["rollout_state_range"][jnp.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])]
        return lower_lim, upper_lim

    def clip_state_b(self, state: State) -> State:
        lower_limit, upper_limit = self.state_b_lim(state)
        return jnp.clip(state, lower_limit, upper_limit)

    def state_b_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """车身坐标系下的状态约束"""
        lower_lim = self.params["rollout_state_b_range"][jnp.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])]
        upper_lim = self.params["rollout_state_b_range"][jnp.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])]
        return lower_lim, upper_lim

    @override
    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.array([-1., -7.])[None, :].repeat(self.num_agents, axis=0) # ax: m/s^2, δ: °
        upper_lim = jnp.array([2., 7.])[None, :].repeat(self.num_agents, axis=0)
        return lower_lim, upper_lim
