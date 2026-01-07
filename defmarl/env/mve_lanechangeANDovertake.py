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
from .designed_scene_gen import gen_scene_randomly, gen_handmade_scene_randomly
from .utils import process_lane_centers, process_lane_marks, relative_state
from ..trainer.data import Rollout
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Reward, Cost, Array, State, AgentState, ObstState, Done, Info
from ..utils.utils import tree_index, MutablePatchCollection, save_anim, calc_2d_rot_matrix, find_closest_goal_indices, \
    gen_i_j_pairs, gen_i_j_pairs_no_identical, normalize_angle
from ..utils.scaling import scaling_calc, scaling_calc_bound

INF = jnp.inf

class MVELaneChangeAndOverTake(MVE):
    """该任务使用agent位姿和预设轨迹的偏移量、加减速度和方向盘转角的大小作为的reward的度量，
    scaling factor作为cost的度量，每个agent分配一个goal并规划出一条轨迹（五次多项式），
    环境为四车道，障碍车均沿车道作匀速直线运动"""

    PARAMS = {
        # 宝骏E300参数，只有bb和m是准的，其它的都是估计的
        "ego_lf": 0.8475, # m，假设质心位于几何中心
        "ego_lr": 0.9025, # m，假设质心位于几何中心
        "ego_bb_size": jnp.array([2.625, 1.647]), # bounding box的[width, height] m
        "ego_m": 940., # kg
        "ego_Iz": 752.25333, # kg*m^2，假设质心位于几何中心
        "ego_Cf": 47850., # N/rad
        "ego_Cr": 46510., # N/rad
        "comm_radius": 100,
        "obst_bb_size": jnp.array([4., 2.]), # bounding box的[width, height] m

        # [x_l, x_h, y_l, y_h, vx_l, vx_h, vy_l, vy_h, θ_l, θ_h, dθdt_l, dθdt_h, \
        # bbw_l, bbw_h, bbh_l, bbh_h]
        # 单位：x,y,bbw,bbh: m  vx,vy: km/h,  θ: °, dθdt: °/s
        # 速度约束通过车身坐标系对纵向速度约束来进行
        "default_state_range": jnp.array([-100., 100., -4.5, 4.5, -INF, INF, -INF, INF, -180., 180., -INF, INF,
        -INF, INF, -INF, INF]), # 默认范围，用于指示正常工作的状态范围
        "rollout_state_range": jnp.array([-120., 220., -10., 10., -INF, INF, -INF, INF, -180., 180., -INF, INF,
        -INF, INF, -INF, INF]), # rollout过程中的限制，强制约束
        "rollout_state_b_range": jnp.array([-INF, INF, -INF, INF, 20., 100., -INF, INF, -INF, INF, -INF, INF,
        -INF, INF, -INF, INF]), # rollout过程中在车身坐标系下状态约束，主要对纵向速度有约束，动力学模型不允许倒车
        "agent_init_state_range": jnp.array([-100., -50., -3., 3., -INF, INF, -INF, INF, -180., 180., -INF, INF,
        -INF, INF, -INF, INF]), # 用于agent初始化的状态范围
        "terminal_state_range": jnp.array([50., 100., -3., 3., -INF, INF, -INF, INF, -180., 180., -INF, INF,
        -INF, INF, -INF, INF]), # 随机生成terminal时的状态范围
        # "obst_state_range": jnp.array([-15., 20., -7., 7.5, 0., 360., 0., 0.]), # 随机生成obstacle的状态范围

        "lane_width": 3, # 车道宽度，m
        "v_bias": 5, # 可允许的速度偏移量
        "alpha_thresh": 1.3, # alpha大于thresh时才判定为安全，用于避障时让agent离obst不要那么近
    }
    PARAMS.update({
        "ego_radius": jnp.linalg.norm(PARAMS["ego_bb_size"]/2), # m
        "ego_L": PARAMS["ego_lf"]+PARAMS["ego_lr"], # m
        "lane_centers": process_lane_centers(PARAMS["default_state_range"][2:4], PARAMS["lane_width"]), # 车道中心线y坐标 m
    })
    if "obst_bb_size" in PARAMS and PARAMS["obst_bb_size"].shape == (2,):
        PARAMS.update({"obst_radius": jnp.linalg.norm(PARAMS["obst_bb_size"]/2)})
    # PARAMS.update({"n_obsts": PARAMS["lane_centers"].shape[0]}) # 本环境每根车道一辆障碍车
    assert PARAMS["terminal_state_range"][0] - PARAMS["agent_init_state_range"][1] >= 100

    def __init__(self,
                 num_agents: int,
                 area_size: Optional[float] = None,
                 max_step: int = 512,
                 max_travel: Optional[float] = None,
                 dt: float = 0.025,
                 params: dict = None
                 ):
        area_size = MVELaneChangeAndOverTake.PARAMS["rollout_state_range"][:4] if area_size is None else area_size
        params = MVELaneChangeAndOverTake.PARAMS if params is None else params
        super(MVELaneChangeAndOverTake, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        # assert self.params["n_obsts"] == MVELaneChangeAndOverTake.PARAMS["n_obsts"], "本环境只接受2个障碍物的设置！"
        self.all_goals = jnp.zeros((num_agents, self.num_goals, self.state_dim))  # 参考点初始化
        self.num_obsts = 0 # 初始化

    @override
    @property
    def state_dim(self) -> int:
        return 8 # x y vx vy θ dθ/dt bw bh

    @override
    @property
    def node_dim(self) -> int:
        return 11  # state_dim(8)  indicator(3): agent: 001, goal: 010, obstacle: 100, pad: 00-1

    @override
    @property
    def edge_dim(self) -> int:
        return 8 # Δstate: Δx, Δy, Δvx, Δvy, Δθ, Δdθ/dt, Δbw, Δbh

    @override
    @property
    def action_dim(self) -> int:
        return 2  # a：车辆纵向加速度（m/s^2） δ：前轮转角（逆时针为正，°）

    @override
    @property
    def reward_max(self):
        return 0.5
        # return 500 # debug

    @property
    def reward_min(self) -> float:
        return -17 # TODO: fine-tune
        # return 300 # debug

    @override
    @property
    def n_cost(self) -> int:
        return 4 # agent间碰撞(1) + agent-obstacle碰撞(1) + agent超出y轴范围(高+低，2)

    @override
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "bound exceeds y low", "bound exceeds y high"

    @property
    def num_goals(self) -> int:
        return 3200 # 每个agent参考轨迹点的数量

    @override
    def reset(self, key: Array) -> GraphsTuple:
        """使用场景类别生成函数进行agent、goal和obstacle的生成"""
        c_ycs = self.params["lane_centers"]
        xrange = self.params["default_state_range"][:2]
        yrange = self.params["default_state_range"][2:4]
        lanewidth = self.params["lane_width"]
        agents, obsts, all_goals = gen_scene_randomly(key, self.num_agents, self.num_goals, xrange, yrange,
                                                             lanewidth, c_ycs)
        self.all_goals = all_goals
        goals_init_indices = find_closest_goal_indices(agents, all_goals)
        agents_indices = jnp.arange(agents.shape[0])
        goals = all_goals[agents_indices, goals_init_indices, :]
        env_state = MVEEnvState(agents, goals, obsts)
        self.num_obsts = obsts.shape[0]

        return self.get_graph(env_state)

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
        as_S_new_unclip = as_S_new_unclip.at[:, 4].set(normalize_angle(as_S_new_unclip[:, 4]))  # θ限制在[-180, 180]°
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

    def goal_step(self, aS_agent_states_new: AgentState) -> State:
        """根据下一步的agent位置，寻找相应的距离最近的目标点作为参考"""
        a_goals_indices = find_closest_goal_indices(aS_agent_states_new, self.all_goals)
        a_agents_indices = jnp.arange(aS_agent_states_new.shape[0])
        aS_goal_states = self.all_goals[a_agents_indices, a_goals_indices, :]

        return aS_goal_states

    @override
    def step(
            self, graph: MVEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MVEEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=MVE.GOAL, n_type=self.num_agents)
        obst_states = graph.type_states(type_idx=MVE.OBST, n_type=self.num_obsts)
        next_obst_states = self.obst_step_euler(obst_states)

        # calculate next graph
        action = self.transform_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_goal_states = self.goal_step(next_agent_states)
        # goal = next_goal_states[:, :2]
        next_env_state = MVEEnvState(next_agent_states, next_goal_states, next_obst_states)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        """
        # debug
        jax.debug.print("============================= \n"
                        "old_states: \n"
                        "agent={old_agent_states} \n"
                        "goal={old_goal_states} \n"
                        "obstacle={old_obstacle_states} \n"
                        "\n"
                        "action={action} \n"
                        "\n"
                        "new_states: \n"
                        "agent={new_agent_states} \n"
                        "goal={new_goal_states} \n"
                        "obstacle={new_obstacle_states} \n"
                        "============================= \n",
                        old_agent_states = agent_states,
                        old_goal_states = goal_states,
                        old_obstacle_states = obst_states,
                        action=action,
                        new_agent_states = next_agent_states,
                        new_goal_states = next_goal_states,
                        new_obstacle_states = next_obst_states)
        """


        return self.get_graph(next_env_state), reward, cost, done, info

    def get_reward(self, graph: MVEEnvGraphsTuple, ad_action: Action) -> Reward:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals

        aS_agents_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        aS_goals_states = graph.type_states(type_idx=MVE.GOAL, n_type=num_goals)
        # state: x, y, vx, vy, θ, dθ/dt, bw, bh
        # 参数提取
        a2_goal_pos_m = aS_goals_states[:, :2]
        a2_goal_v_kmph = aS_goals_states[:, 2:4]
        a_goal_theta_deg = aS_goals_states[:, 4]
        a2_agent_pos_m = aS_agents_states[:, :2]
        a2_agent_v_kmph = aS_agents_states[:, 2:4]
        a_agent_theta_deg = aS_agents_states[:, 4]

        # 旋转矩阵计算
        a22_Q_goal = jax.vmap(calc_2d_rot_matrix, in_axes=(0))(a_goal_theta_deg)
        a22_Q_agent = jax.vmap(calc_2d_rot_matrix, in_axes=(0))(a_agent_theta_deg)

        # 自车坐标系下的横纵向速度计算
        a2_goal_v_b_kmph = jnp.einsum('aij, ai -> aj', a22_Q_goal, a2_goal_v_kmph)
        a2_agent_v_b_kmph = jnp.einsum('aij, ai -> aj', a22_Q_agent, a2_agent_v_kmph)

        reward = jnp.zeros(()).astype(jnp.float32)
        # 循迹奖励： 位置+角度
        # 位置奖励，和目标点的欧氏距离
        a_dist = jnp.linalg.norm(a2_goal_pos_m - a2_agent_pos_m, axis=1)
        reward -= a_dist.mean() * 0.01

        # 角度奖励
        a_costheta_dist = jnp.cos((a_goal_theta_deg - a_agent_theta_deg) * jnp.pi/180)
        reward += (a_costheta_dist.mean() - 1) * 0.001

        # 速度跟踪惩罚
        a_delta_v = a2_goal_v_b_kmph[:, 0] - a2_agent_v_b_kmph[:, 0]
        # reward -= (a_delta_v**2).mean() * 0.00005 # 只比较x方向（纵向）速度
        reward -= jnp.abs(a_delta_v).mean() * 0.0005
        reward -= jnp.where(jnp.abs(a_delta_v) > self.params["v_bias"], 1., 0.).mean() * 0.005

        # 动作惩罚
        reward -= (ad_action[:, 0]**2).mean() * 0.00005
        reward -= (ad_action[:, 1]**2).mean() * 0.0001

        return reward

    def get_cost(self, graph: MVEEnvGraphsTuple) -> Cost:
        """使用射线法计算的scaling factor：α为cost的评判指标，thresh-α<0安全，>=0不安全"""
        thresh = self.params["alpha_thresh"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]

        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        # agent之间的scaling factor
        """
        if num_agents == 1:
            a_agent_cost = -jnp.ones((1,), dtype=jnp.float32)
        else :
            i_pairs, j_pairs = gen_i_j_pairs_no_identical(num_agents, num_agents)
            state_i_pairs = agent_states[i_pairs, :]
            state_j_pairs = agent_states[j_pairs, :]
            alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
            alpha_matrix = jnp.full((num_agents, num_agents), INF)  # 初始化矩阵，填充无穷大
            alpha_matrix = alpha_matrix.at[i_pairs, j_pairs].set(alpha_pairs)
            # 每个agent对应的行取最大值（即与其他agent的最小α，α越小越不安全）
            a_agent_cost = jnp.max(thresh-alpha_matrix, axis=1)
        """
        a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug

        # agent 和 obst 之间的scaling factor
        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
        else:
            obstacle_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            state_i_pairs = agent_states[i_pairs, :]
            state_j_pairs = obstacle_states[j_pairs, :]
            alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
            alpha_matrix = alpha_pairs.reshape((num_agents, num_obsts))
            a_obst_cost = jnp.max(thresh-alpha_matrix, axis=1)
        # a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug

        # agent 和 bound 之间的scaling factor，只对y方向有约束
        state_range = self.params["default_state_range"]
        yl = state_range[2]
        A = jnp.array([[0., 1.]])
        b = jnp.array([yl])
        a_bound_yl_cost = thresh - jax.vmap(scaling_calc_bound, in_axes=(0, None, None))(agent_states, A, b)

        yh = state_range[3]
        A = jnp.array([[0., -1.]])
        b = jnp.array([-yh])
        a_bound_yh_cost = thresh - jax.vmap(scaling_calc_bound, in_axes=(0, None, None))(agent_states, A, b)

        # a_bound_yl_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug
        # a_bound_yh_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug

        cost = jnp.stack([a_agent_cost, a_obst_cost, a_bound_yl_cost, a_bound_yh_cost], axis=1)
        assert cost.shape == (num_agents, self.n_cost)

        """
        # debug
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

        # add margin and clip
        eps = 1.
        cost = jnp.where(cost <= 0.0, cost, cost + eps)
        cost = jnp.clip(cost, a_min=-3.0)

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
        T_goal_states = jax.vmap(lambda x: x.type_states(type_idx=MVE.GOAL, n_type=self.num_agents))(rollout.graph)
        ref_goals = T_goal_states[:, :, :2]
        n_goals = self.num_agents if n_goals is None else n_goals

        ax: Axes
        xlim = self.params["rollout_state_range"][:2]
        ylim = self.params["default_state_range"][2:4]
        fig, ax = plt.subplots(1, 1, figsize=(30,
                                (ylim[1]+3-(ylim[0]-3))*20/(xlim[1]+3-(xlim[0]-3))+4)
                               , dpi=100)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0]-3, ylim[1]+3)
        ax.set(aspect="equal")
        plt.axis("on")
        if viz_opts is None:
            viz_opts = {}

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
        obsts_state = graph0.type_nodes(type_idx=MVE.OBST, n_type=self.num_obsts)
        # state: x, y, vx, vy, θ, dθ/dt, bw, bh
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

        # plot reference points
        # state: x, y, vx, vy, θ, dθ/dt, bw,
        all_ref_xs = ref_goals[:, :, 0].reshape(-1)
        all_ref_ys = ref_goals[:, :, 1].reshape(-1)
        ax.scatter(all_ref_xs, all_ref_ys, color=goal_color, zorder=7, s=5, alpha=1.0, marker='.')

        # plot edges
        all_pos = graph0.states[:, :2]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents + n_goals + self.num_obsts, axis=0)
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
            alpha=0.
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
            for ii in range(self.num_obsts):
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
            is_pad_t = np.any(e_edge_index_t == self.num_agents + n_goals + self.num_obsts, axis=0)
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
        agent_agent_edges = []
        if num_agents > 1:
            pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
            dist = jnp.linalg.norm(pos_diff, axis=-1)
            dist += jnp.eye(dist.shape[1]) * (self.params["comm_radius"] + 1)
            agent_agent_mask = jnp.less(dist, self.params["comm_radius"])
            i_pairs, j_pairs = gen_i_j_pairs_no_identical(num_agents, num_agents)
            agent_state_i_pairs = state.agent[i_pairs, :]
            agent_state_j_pairs = state.agent[j_pairs, :]
            rel_state_pairs = jax.vmap(relative_state, in_axes=(0, 0))(agent_state_i_pairs, agent_state_j_pairs)
            rel_state = jnp.zeros((num_agents, num_agents, self.state_dim), dtype=jnp.float32) # 相对状态矩阵初始化
            rel_state = rel_state.at[i_pairs, j_pairs, :].set(rel_state_pairs)
            agent_agent_edges = [EdgeBlock(rel_state, agent_agent_mask, id_agent, id_agent)]
        """

        # agent - goal connection
        agent_goal_edges = []
        for i_agent in range(self.num_agents):
            agent_state_i = state.agent[i_agent]
            goal_state_i = state.goal[i_agent]
            rel_state = relative_state(agent_state_i, goal_state_i)
            agent_goal_edges.append(EdgeBlock(rel_state[None, None, :], jnp.ones((1, 1)),
                                              jnp.array([i_agent]), jnp.array([i_agent + self.num_agents])))

        # agent - obstacle connection
        agent_obst_edges = []
        if num_obsts > 0:
            obs_pos = state.obstacle[:, :2]
            poss_diff = agent_pos[:, None, :] - obs_pos[None, :, :]
            dist = jnp.linalg.norm(poss_diff, axis=-1)
            agent_obs_mask = jnp.less(dist, self.params["comm_radius"])
            id_obs = jnp.arange(num_obsts) + num_agents * 2
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            agent_state_i_pairs = state.agent[i_pairs, :]
            obst_state_j_pairs = state.obstacle[j_pairs, :]
            rel_state_pairs = jax.vmap(relative_state, in_axes=(0, 0))(agent_state_i_pairs, obst_state_j_pairs)
            rel_state = rel_state_pairs.reshape((num_agents, num_obsts, self.state_dim))
            agent_obst_edges = [EdgeBlock(rel_state, agent_obs_mask, id_agent, id_obs)]

        # return agent_agent_edges + agent_goal_edges + agent_obst_edges

        """
        #debug
        jax.debug.print("=============================== \n"
                        "agent_goal_rel_state = {rel_state} \n"
                        "agent_goal_mask = {agent_goal_mask} \n"
                        "=============================== \n",
                        rel_state=rel_state,
                        agent_goal_mask=agent_goal_mask)
        """

        return agent_goal_edges + agent_obst_edges # 跟踪任务debug

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
        # state: x y vx vy θ dθdt bw bh
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

        # node type
        node_type = -jnp.ones((num_agents + num_goals + num_obsts), dtype=jnp.int32)
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[num_agents: num_agents + num_goals].set(MVE.GOAL)
        if num_obsts > 0:
            node_type = node_type.at[num_agents + num_goals:].set(MVE.OBST)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
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
        lower_lim = self.params["rollout_state_range"][jnp.array([0, 2, 4, 6, 8, 10, 12, 14])]
        upper_lim = self.params["rollout_state_range"][jnp.array([1, 3, 5, 7, 9, 11, 13, 15])]
        return lower_lim, upper_lim

    def clip_state_b(self, state: State) -> State:
        lower_limit, upper_limit = self.state_b_lim(state)
        return jnp.clip(state, lower_limit, upper_limit)

    def state_b_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """车身坐标系下的状态约束"""
        lower_lim = self.params["rollout_state_b_range"][jnp.array([0, 2, 4, 6, 8, 10, 12, 14])]
        upper_lim = self.params["rollout_state_b_range"][jnp.array([1, 3, 5, 7, 9, 11, 13, 15])]
        return lower_lim, upper_lim

    @override
    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.array([-1., -7.])[None, :].repeat(self.num_agents, axis=0) # ax: m/s^2, δ: °
        upper_lim = jnp.array([2., 7.])[None, :].repeat(self.num_agents, axis=0)
        return lower_lim, upper_lim

    def ros_run(self):
        """在reset初始化之后才能使用，可能还需要一个ros_init函数进行ros节点的初始化？"""

        # 从action节点订阅action信息

        # step，包含agent、goal、obst的step，考虑添加一定的时延？

        # 发布状态信息至env节点