import jax
import jax.numpy as jnp
import jax.random as jr

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

from ..utils.typing import PRNGKey, State, AgentState, ObstState, Array, PathRefs
from ..utils.utils import calc_linear_eff, calc_quintic_eff, const_f, linear_f, quintic_polynomial_f, three_sec_f

def generate_lanechange_path_points(xrange: Array,
                                    num_agents: int,
                                    num_points: int,
                                    S_start_state: AgentState,
                                    S_terminal_state: AgentState,
                                    points_interval: int=0.1) -> PathRefs:
    """生成由水平直线-五次多项式曲线-水平直线组成的分段参考轨迹点，默认每0.1m生成一个参考点，共生成3200个"""
    # 生成中间的五次多项式
    one6_patheffs_f, one6_patheffs_df, one6_patheffs_ddf = calc_quintic_eff(S_start_state[None,:], S_terminal_state[None,:])
    quintic_f = quintic_polynomial_f(one6_patheffs_f)
    quintic_df = quintic_polynomial_f(one6_patheffs_df)
    quintic_ddf = quintic_polynomial_f(one6_patheffs_ddf)
    # 构建三个值的常数函数
    zeros = jnp.zeros((1, 1), dtype=jnp.float32)
    const_f_ystart = const_f(S_start_state[1][None, None])
    const_f_yterminal = const_f(S_terminal_state[1][None, None])
    const_f_zeros = const_f(zeros)
    # 构建中间为五次多项式的分段函数
    poly_sec_f = three_sec_f(const_f_ystart, quintic_f, const_f_yterminal,
                             S_start_state[0][None, None], S_terminal_state[0][None, None])
    poly_sec_df = three_sec_f(const_f_zeros, quintic_df, const_f_zeros,
                              S_start_state[0][None, None], S_terminal_state[0][None, None])
    poly_sec_ddf = three_sec_f(const_f_zeros, quintic_ddf, const_f_zeros,
                               S_start_state[0][None, None], S_terminal_state[0][None, None])
    # 构建路径点
    onen_xs = jnp.linspace(start=xrange[0][None], stop=xrange[0][None] + (num_points + 1) * points_interval, num=num_points,
                         dtype=jnp.float32).T
    onen_ys = poly_sec_f(onen_xs)
    onen_dys = poly_sec_df(onen_xs)
    onen_ddys = poly_sec_ddf(onen_xs)
    onen_thetas_rad = jnp.arctan(onen_dys)
    onen_thetas_deg = onen_thetas_rad * 180 / jnp.pi
    # state: x y vx vy θ dθdt bw bh
    onen_vs_kmph = jnp.repeat(S_terminal_state[2][None, None], onen_thetas_rad.shape[1], axis=1)
    onen_vxs_kmph = onen_vs_kmph * jnp.cos(onen_thetas_rad)
    onen_vys_kmph = onen_vs_kmph * jnp.sin(onen_thetas_rad)
    onen_dthetas_radps = onen_ddys * onen_vxs_kmph / 3.6 / (1 + onen_dys ** 2)
    onen_dthetas_degps = onen_dthetas_radps * 180 / jnp.pi
    onen_zeros = jnp.zeros_like(onen_xs)

    onenS_goals = jnp.stack([onen_xs, onen_ys, onen_vxs_kmph, onen_vys_kmph, onen_thetas_deg, onen_dthetas_degps, onen_zeros, onen_zeros],
                          axis=2)
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)
    return anS_goals

def generate_horizontal_path_points(xrange: Array,
                                    num_agents: int,
                                    num_points: int,
                                    start_y: jnp.ndarray, # shape应为()
                                    terminal_vx: jnp.ndarray, # shape应为()
                                    points_interval: int = 0.1) -> PathRefs:
    """生成由水平直线参考轨迹点，默认每0.1m生成一个参考点，共生成3200个"""
    assert start_y.shape == terminal_vx.shape == ()
    # 构建路径点
    onen_xs = jnp.linspace(start=xrange[0][None], stop=xrange[0][None] + (num_points + 1) * points_interval, num=num_points,
                         dtype=jnp.float32).T
    onen_ys = jnp.repeat(start_y[None, None], num_points, axis=1)
    onen_thetas_rad = jnp.zeros_like(onen_ys)
    onen_thetas_deg = jnp.zeros_like(onen_ys)
    # state: x y vx vy θ dθdt bw bh
    onen_vs_kmph = jnp.repeat(terminal_vx[None, None], num_points, axis=1)
    onen_vxs_kmph = onen_vs_kmph
    onen_vys_kmph = jnp.zeros_like(onen_ys)
    onen_dthetas_degps = jnp.zeros_like(onen_ys)
    onen_zeros = jnp.zeros_like(onen_xs)

    onenS_goals = jnp.stack([onen_xs, onen_ys, onen_vxs_kmph, onen_vys_kmph, onen_thetas_deg, onen_dthetas_degps, onen_zeros, onen_zeros],
                          axis=2)
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)
    return anS_goals


class SceneBase(ABC):
    """用于生成一些相对固定的基础场景"""

    def __init__(self, key:PRNGKey, num_agents:int):
        self.key = key
        self.num_agents = num_agents

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_ref_points(self) -> int:
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def num_lanes(self) -> int:
        pass

    @property
    @abstractmethod
    def lane_centers(self) -> Array:
        pass

    @property
    @abstractmethod
    def xrange(self) -> Array:
        pass

    @property
    @abstractmethod
    def yrange(self) -> Array:
        pass

    @property
    @abstractmethod
    def num_moving_obsts(self) -> int:
        pass

    @property
    @abstractmethod
    def num_static_obsts(self) -> int:
        pass

    @property
    def num_obsts(self) -> int:
        return self.num_moving_obsts + self.num_static_obsts

    @abstractmethod
    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        pass


class LaneChangeANDOvertakeScene(SceneBase, ABC):
    """专用于lanechangeANDovertake任务的场景，需要指定x和y的范围以及车道宽度，所有agent共享同一条轨迹，但起点和初始状态不一定一样"""

    def __init__(self, key:PRNGKey, num_agents:int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(LaneChangeANDOvertakeScene, self).__init__(key, num_agents)
        self._xrange = xrange
        self._yrange = yrange
        self._num_ref_points = num_ref_points
        self._lane_width = lane_width
        self._lane_centers = lane_centers

    @property
    def num_ref_points(self) -> int:
        return self._num_ref_points

    @property
    def state_dim(self) -> int:
        return 8 # state: x y vx vy θ dθ/dt bw bh

    @property
    def num_lanes(self):
        return self._lane_centers.shape[0]

    @property
    def lane_centers(self):
        return self._lane_centers

    @property
    def xrange(self):
        return self._xrange

    @property
    def yrange(self):
        return self._yrange


class LaneChangeMiddleStaticEdgeFastMoving(LaneChangeANDOvertakeScene):
    """此场景只可适用于3车道场景，可布置1个动态障碍物，1个静态障碍物，静态障碍物放在中间车道，动态障碍物放在第3/1车道，同时动态障碍物车速较快，
    agent由第1/3车道向第3/1车道变道，使得agent在差不多到第3/1车道时与动态障碍物发生碰撞，图例如下：
    ==================================================
    1   □ ---------> moving obstacle/-------------> reference path
    -----------------------------/--------------------
    2                         ♦  static obstacle
    ------------------------/-------------------------
    3     ego  ■ --------/
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(LaneChangeMiddleStaticEdgeFastMoving, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'lanechange_scene_with_middle_static_obstacle_and_edge_fast_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_vx_key, agent_x_key, agent_y_key, agent_vx_key, \
            sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 12)
        num_lanes = self.num_lanes

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1]-self.xrange[0])/3+self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2*(self.xrange[1]-self.xrange[0])/3+self.xrange[0],
                                maxval=self.xrange[1])
        start_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0,2])], shape=())
        terminal_y = self.lane_centers[1] - (start_y - self.lane_centers[1])
        start_vx = terminal_vx = jr.uniform(start_terminal_vx_key, shape=(), dtype=jnp.float32,
                                            minval=60, maxval=90) # 60 ~ 90 km/h
        Sm3_other0 = jnp.zeros((self.state_dim-3,), dtype=jnp.float32)
        S_start_state = jnp.concatenate([start_x[None], start_y[None], start_vx[None], Sm3_other0])
        S_terminal_state = jnp.concatenate([terminal_x[None], terminal_y[None], terminal_vx[None], Sm3_other0])
        anS_goals = generate_lanechange_path_points(self.xrange, self.num_agents, self.num_ref_points, S_start_state,
                                                    S_terminal_state)

        # 生成初始agent，x坐标都一样，y和vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0) # 变道前同一x
        a_agent_y = jr.choice(agent_y_key, self.lane_centers, shape=(self.num_agents,)) # 几根车道中选
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=60, maxval=90) # 60 ~ 90 km/h
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0], axis=1)

        # 生成静态障碍物，x坐标位于变道多项式中间，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst_x = (start_x + terminal_x) / 2
        sobst_y = jr.uniform(sobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., sobst_theta, 0., 0., 0.])

        # 生成动态障碍物，y坐标位于变道目标车道坐标，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好变道到目标车道时的时间
        t = (terminal_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120) # 90 ~ 120 km/h
        mobst_x = terminal_x - t * mobst_vx - 5 + \
            jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-2, maxval=2)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class LaneChangeMiddleStaticEdgeSlowMoving(LaneChangeANDOvertakeScene):
    """此场景只可适用于3车道场景，可布置1个动态障碍物，1个静态障碍物，静态障碍物放在中间车道，动态障碍物放在第3/1车道，同时动态障碍物车速较慢，
    agent由第1/3车道向第3/1车道变道，使得agent在变道完成后短时间内需要超越运动较慢的动态障碍物以避免碰撞，图例如下：
    ==================================================
    1                               /------ □ ----> moving obstacle
    -----------------------------/--------------------
    2                         ♦  static obstacle
    ------------------------/-------------------------
    3     ego  ■ --------/ reference path
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(LaneChangeMiddleStaticEdgeSlowMoving, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'lanechange_scene_with_middle_static_obstacle_and_edge_slow_moving_scene'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_vx_key, agent_x_key, agent_y_key, agent_vx_key, \
            sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 12)
        num_lanes = self.num_lanes

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1]-self.xrange[0])/3+self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2*(self.xrange[1]-self.xrange[0])/3+self.xrange[0],
                                maxval=self.xrange[1])
        start_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0,2])], shape=())
        terminal_y = self.lane_centers[1] - (start_y - self.lane_centers[1])
        start_vx = terminal_vx = jr.uniform(start_terminal_vx_key, shape=(), dtype=jnp.float32,
                                            minval=60, maxval=90) # 60 ~ 90 km/h
        Sm3_other0 = jnp.zeros((self.state_dim-3,), dtype=jnp.float32)
        S_start_state = jnp.concatenate([start_x[None], start_y[None], start_vx[None], Sm3_other0])
        S_terminal_state = jnp.concatenate([terminal_x[None], terminal_y[None], terminal_vx[None], Sm3_other0])
        anS_goals = generate_lanechange_path_points(self.xrange, self.num_agents, self.num_ref_points, S_start_state,
                                                    S_terminal_state)

        # 生成初始agent，x坐标都一样，y和vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0) # 变道前同一x
        a_agent_y = jr.choice(agent_y_key, self.lane_centers, shape=(self.num_agents,)) # 几根车道中选
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=60, maxval=90) # 60 ~ 90 km/h
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0], axis=1)

        # 生成静态障碍物，x坐标位于变道多项式中间，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst_x = (start_x + terminal_x) / 2
        sobst_y = jr.uniform(sobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., sobst_theta, 0., 0., 0.])

        # 生成动态障碍物，y坐标位于变道目标车道坐标，可稍有浮动，vx可随机选择一个较较小值，x坐标需要计算agent恰好变道到目标车道时动态障碍物位于
        # agent之前
        t = (terminal_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=20, maxval=50) # 20 ~ 50 km/h
        mobst_x = terminal_x - t * mobst_vx  + \
            jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-10, maxval=10)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class LaneChangeEdgeStaticMiddleFastMoving(LaneChangeANDOvertakeScene):
    """此场景只可适用于3车道场景，可布置1个动态障碍物，1个静态障碍物，静态障碍物放在第1/3车道，动态障碍物放在中间车道，同时动态障碍物车速较快，
    agent由第3/1车道向第1/3车道变道，使得agent在变道至中间车道时差不多需要避免与动态障碍物的碰撞，图例如下：
    ==================================================
    1                                /----♦  static obstacle--> reference path
    ---------------------------- -/--------------------
    2      □ ------------------/-> moving obstacle
    ------------------------/-------------------------
    3     ego  ■ --------/
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(LaneChangeEdgeStaticMiddleFastMoving, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'lanechange_scene_with_edge_static_obstacle_and_middle_fast_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_vx_key, agent_x_key, agent_y_key, agent_vx_key, \
            sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 12)
        num_lanes = self.num_lanes

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 2])], shape=())
        terminal_y = self.lane_centers[1] - (start_y - self.lane_centers[1])
        start_vx = terminal_vx = jr.uniform(start_terminal_vx_key, shape=(), dtype=jnp.float32,
                                            minval=60, maxval=90)  # 60 ~ 90 km/h
        Sm3_other0 = jnp.zeros((self.state_dim - 3,), dtype=jnp.float32)
        S_start_state = jnp.concatenate([start_x[None], start_y[None], start_vx[None], Sm3_other0])
        S_terminal_state = jnp.concatenate([terminal_x[None], terminal_y[None], terminal_vx[None], Sm3_other0])
        anS_goals = generate_lanechange_path_points(self.xrange, self.num_agents, self.num_ref_points, S_start_state,
                                                    S_terminal_state)
        aS_goal_init = anS_goals[:, 0, :]

        # 生成初始agent，x坐标都一样，y和vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)  # 变道前同一x
        a_agent_y = jr.choice(agent_y_key, self.lane_centers, shape=(self.num_agents,))  # 几根车道中选
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=60, maxval=90)  # 60 ~ 90 km/h
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0],
                                         axis=1)

        # 生成静态障碍物，位于变道结束位置，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst_x = terminal_x - 5
        sobst_y = jr.uniform(sobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., sobst_theta, 0., 0., 0.])

        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好变道到中间车道时动态障碍物位于
        # agent附近
        t = ((start_x + terminal_x) / 2 - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120)  # 90 ~ 120 km/h
        mobst_x = (start_x + terminal_x) / 2 - t * mobst_vx - 10 + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-5, maxval=5)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class LaneChangeEdgeStaticMiddleSlowMoving(LaneChangeANDOvertakeScene):
    """此场景只可适用于3车道场景，可布置1个动态障碍物，1个静态障碍物，静态障碍物放在第1/3车道，动态障碍物放在中间车道，同时动态障碍物车速较慢，
    agent由第3/1车道向第1/3车道变道，使得agent在变道至中间车道时差不多需要避免与动态障碍物的碰撞，图例如下：
    ==================================================
    1                                /----♦  static obstacle--> reference path
    ------------------------------/--------------------
    2  □ --moving obstacle-->  /
    ------------------------/-------------------------
    3     ego  ■ --------/ reference path
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(LaneChangeEdgeStaticMiddleSlowMoving, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'lanechange_scene_with_edge_static_obstacle_and_middle_slow_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_vx_key, agent_x_key, agent_y_key, agent_vx_key, \
            sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 12)
        num_lanes = self.num_lanes

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 2])], shape=())
        terminal_y = self.lane_centers[1] - (start_y - self.lane_centers[1])
        start_vx = terminal_vx = jr.uniform(start_terminal_vx_key, shape=(), dtype=jnp.float32,
                                            minval=60, maxval=90)  # 60 ~ 90 km/h
        Sm3_other0 = jnp.zeros((self.state_dim - 3,), dtype=jnp.float32)
        S_start_state = jnp.concatenate([start_x[None], start_y[None], start_vx[None], Sm3_other0])
        S_terminal_state = jnp.concatenate([terminal_x[None], terminal_y[None], terminal_vx[None], Sm3_other0])
        anS_goals = generate_lanechange_path_points(self.xrange, self.num_agents, self.num_ref_points, S_start_state,
                                                    S_terminal_state)
        aS_goal_init = anS_goals[:, 0, :]

        # 生成初始agent，x坐标都一样，y和vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)  # 变道前同一x
        a_agent_y = jr.choice(agent_y_key, self.lane_centers, shape=(self.num_agents,))  # 几根车道中选
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=60, maxval=90)  # 60 ~ 90 km/h
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0],
                                         axis=1)

        # 生成静态障碍物，位于变道结束位置，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst_x = terminal_x - 5
        sobst_y = jr.uniform(sobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., sobst_theta, 0., 0., 0.])

        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较小值，x坐标需要计算agent恰好变道到中间车道时动态障碍物位于
        # agent附近
        t = ((start_x + terminal_x) / 2 - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=20, maxval=50)  # 20 ~ 50 km/h
        mobst_x = (start_x + terminal_x) / 2 - t * mobst_vx - 5 + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-5, maxval=5)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class OvertakeEdgeStaticMiddleFastMoving(LaneChangeANDOvertakeScene):
    """此场景只可适用于3车道场景，可布置1个动态障碍物，1个静态障碍物，静态障碍物放在第1/3车道，动态障碍物放在中间车道，同时动态障碍物车速较快，
    agent沿第1/3车道直行，使得agent在差不多需要避让静态障碍物时也要避免与动态障碍物的碰撞，图例如下：
    ==================================================
    1
    ---------------------------------------------------
    2  □ -------------------> moving obstacle
    --------------------------------------------------
    3     ego  ■ --------♦  static obstacle--> reference path
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(OvertakeEdgeStaticMiddleFastMoving, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'overtake_scene_with_edge_static_obstacle_and_middle_fast_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_vx_key, agent_x_key, agent_vx_key, \
            sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 11)
        num_lanes = self.num_lanes

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 2])], shape=())
        terminal_vx = jr.uniform(start_terminal_vx_key, shape=(), dtype=jnp.float32,
                                            minval=60, maxval=90)  # 60 ~ 90 km/h
        Sm3_other0 = jnp.zeros((self.state_dim - 3,), dtype=jnp.float32)
        anS_goals = generate_horizontal_path_points(self.xrange, self.num_agents, self.num_ref_points, start_y,
                                                    terminal_vx)

        # 生成初始agent，x和y坐标都一样，vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)  # 变道前同一x
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0)
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=60, maxval=90)  # 60 ~ 90 km/h
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0],
                                         axis=1)

        # 生成静态障碍物，位于start和terminal中间，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst_x = (start_x + terminal_x) / 2
        sobst_y = jr.uniform(sobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., sobst_theta, 0., 0., 0.])

        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好到静态障碍物时时动态障碍物位于
        # agent附近
        t = (sobst_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120)  # 90 ~ 120 km/h
        mobst_x = sobst_x - t * mobst_vx  + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=5, maxval=15)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class OvertakeEdgeStaticMiddleSlowMoving(LaneChangeANDOvertakeScene):
    """此场景只可适用于3车道场景，可布置1个动态障碍物，1个静态障碍物，静态障碍物放在第1/3车道，动态障碍物放在中间车道，同时动态障碍物车速较慢，
    agent沿第1/3车道直行，使得agent在差不多需要避让静态障碍物时也要避免与动态障碍物的碰撞，图例如下：
    ==================================================
    1
    ---------------------------------------------------
    2  □ --------------> moving obstacle
    --------------------------------------------------
    3     ego  ■ --------♦  static obstacle--> reference path
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(OvertakeEdgeStaticMiddleSlowMoving, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'overtake_scene_with_edge_static_obstacle_and_middle_slow_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_vx_key, agent_x_key, agent_vx_key, \
            sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 11)
        num_lanes = self.num_lanes

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 2])], shape=())
        terminal_vx = jr.uniform(start_terminal_vx_key, shape=(), dtype=jnp.float32,
                                            minval=60, maxval=90)  # 60 ~ 90 km/h
        Sm3_other0 = jnp.zeros((self.state_dim - 3,), dtype=jnp.float32)
        anS_goals = generate_horizontal_path_points(self.xrange, self.num_agents, self.num_ref_points, start_y,
                                                    terminal_vx)
        aS_goal_init = anS_goals[:, 0, :]

        # 生成初始agent，x和y坐标都一样，vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)  # 变道前同一x
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0)
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=60, maxval=90)  # 60 ~ 90 km/h
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0],
                                         axis=1)

        # 生成静态障碍物，位于start和terminal中间，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst_x = (start_x + terminal_x) / 2
        sobst_y = jr.uniform(sobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., sobst_theta, 0., 0., 0.])

        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较小值，x坐标需要计算agent恰好到静态障碍物时时动态障碍物位于
        # agent附近
        t = (sobst_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=20, maxval=50)  # 20 ~ 50 km/h
        mobst_x = sobst_x - t * mobst_vx  + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-5, maxval=5)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class OvertakeEdgeMiddleStaticsEdgeFastMoving(LaneChangeANDOvertakeScene):
    """此场景只可适用于3车道场景，可布置1个动态障碍物，2个静态障碍物，静态障碍物中1个放在第1/3车道，另1个放在中间车道，2个静态障碍物的y坐标差值
    不超过5米，动态障碍物放在第3/1车道，同时动态障碍物车速较快，agent沿中间车道直行，使得agent在差不多需要避让静态障碍物时也要避免与动态障碍物
    的碰撞，图例如下：
    ==================================================
    1 □ ----------------> moving obstacle
    ---------------------------------------------------
    2  ego  ■ --------♦  static obstacle 2--> reference path
    --------------------------------------------------
    3                    ♦  static obstacle 1
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(OvertakeEdgeMiddleStaticsEdgeFastMoving, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'overtake_scene_with_edge_and_static_obstacles_and_edge_fast_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 2

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        start_x_key, terminal_x_key, start_terminal_vx_key, agent_x_key, agent_vx_key, sobst2_x_key, \
            sobst2_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 11)
        num_lanes = self.num_lanes

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_y = terminal_y = self.lane_centers[1] # 中间车道
        terminal_vx = jr.uniform(start_terminal_vx_key, shape=(), dtype=jnp.float32,
                                            minval=70, maxval=80)  # 70 ~ 80 km/h
        Sm3_other0 = jnp.zeros((self.state_dim - 3,), dtype=jnp.float32)
        anS_goals = generate_horizontal_path_points(self.xrange, self.num_agents, self.num_ref_points, start_y,
                                                    terminal_vx)

        # 生成初始agent，x和y坐标都一样，vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)  # 变道前同一x
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0)
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=70, maxval=80)  # 70 ~ 80 km/h
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0],
                                         axis=1)

        # 生成静态障碍物1，位于start和terminal中间，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst1_x = (start_x + terminal_x) / 2
        sobst1_y = terminal_y
        two_sobst_theta = jr.uniform(sobst_theta_key, shape=(2,), dtype=jnp.float32, minval=0, maxval=0)
        sobst1_theta = two_sobst_theta[0]
        S_sobst1_state = jnp.stack([sobst1_x, sobst1_y, 0., 0., sobst1_theta, 0., 0., 0.])

        # 生成静态障碍物2，x坐标位于静态障碍物1附近，y坐标为第1/3车道中心，航向角可选-180°到180°之间
        sobst2_x = sobst1_x + jr.uniform(sobst2_x_key, shape=(), dtype=jnp.float32, minval=-5, maxval=5)
        sobst2_y = jr.choice(sobst2_y_key, self.lane_centers[jnp.array([0, 2])], shape=())
        sobst2_theta = two_sobst_theta[1]
        S_obst2_state = jnp.stack([sobst2_x, sobst2_y, 0., 0., sobst2_theta, 0., 0., 0.])

        # 生成动态障碍物，y坐标位于第3/1车道，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好到静态障碍物1时动态障碍物位于
        # agent附近
        t = (sobst1_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + \
            self.lane_centers[1] - (sobst2_y - self.lane_centers[1])
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120)  # 90 ~ 120 km/h
        mobst_x = sobst1_x - t * mobst_vx  + \
            jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-10, maxval=10)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst1_state, S_obst2_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class OvertakeEdgeMiddleStaticsEdgeSlowMoving(LaneChangeANDOvertakeScene):
    """此场景只可适用于3车道场景，可布置1个动态障碍物，2个静态障碍物，静态障碍物中1个放在第1/3车道，另1个放在中间车道，2个静态障碍物的y坐标差值
    不超过5米，动态障碍物放在第3/1车道，同时动态障碍物车速较慢，agent沿中间车道直行，使得agent在差不多需要避让静态障碍物时也要避免与动态障碍物
    的碰撞，图例如下：
    ==================================================
    1 □ ------------> moving obstacle
    ---------------------------------------------------
    2  ego  ■ --------♦  static obstacle--> reference path
    --------------------------------------------------
    3                    ♦  static obstacle
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(OvertakeEdgeMiddleStaticsEdgeSlowMoving, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'overtake_scene_with_edge_and_static_obstacles_and_edge_slow_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 2

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        start_x_key, terminal_x_key, start_terminal_vx_key, agent_x_key, agent_vx_key, sobst2_x_key, \
            sobst2_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 11)
        num_lanes = self.num_lanes

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_y = terminal_y = self.lane_centers[1] # 中间车道
        terminal_vx = jr.uniform(start_terminal_vx_key, shape=(), dtype=jnp.float32,
                                            minval=70, maxval=80)  # 70 ~ 80 km/h
        Sm3_other0 = jnp.zeros((self.state_dim - 3,), dtype=jnp.float32)
        anS_goals = generate_horizontal_path_points(self.xrange, self.num_agents, self.num_ref_points, start_y,
                                                    terminal_vx)

        # 生成初始agent，x和y坐标都一样，vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)  # 变道前同一x
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0)
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=70, maxval=80)  # 70 ~ 80 km/h
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0],
                                         axis=1)

        # 生成静态障碍物1，位于start和terminal中间，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst1_x = (start_x + terminal_x) / 2
        sobst1_y = terminal_y
        two_sobst_theta = jr.uniform(sobst_theta_key, shape=(2,), dtype=jnp.float32, minval=0, maxval=0)
        sobst1_theta = two_sobst_theta[0]
        S_sobst1_state = jnp.stack([sobst1_x, sobst1_y, 0., 0., sobst1_theta, 0., 0., 0.])

        # 生成静态障碍物2，x坐标位于静态障碍物1附近，y坐标为第1/3车道中心，航向角可选-180°到180°之间
        sobst2_x = sobst1_x + jr.uniform(sobst2_x_key, shape=(), dtype=jnp.float32, minval=-5, maxval=5)
        sobst2_y = jr.choice(sobst2_y_key, self.lane_centers[jnp.array([0, 2])], shape=())
        sobst2_theta = two_sobst_theta[1]
        S_obst2_state = jnp.stack([sobst2_x, sobst2_y, 0., 0., sobst2_theta, 0., 0., 0.])

        # 生成动态障碍物，y坐标位于第3/1车道，可稍有浮动，vx可随机选择一个较小值，x坐标需要计算agent恰好到静态障碍物1时动态障碍物位于
        # agent附近
        t = (sobst1_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + \
            self.lane_centers[1] - (sobst2_y - self.lane_centers[1])
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=20, maxval=50)  # 90 ~ 120 km/h
        mobst_x = sobst1_x - t * mobst_vx  + \
            jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-5, maxval=5)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst1_state, S_obst2_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class OvertakeInLowSpeed(SceneBase):
    """专用于给c8园区做低速双车道超车任务的场景，需要指定x和y的范围以及车道宽度，所有agent共享同一条轨迹，但起点和初始状态不一定一样，图例如下：
    ===============================================================================
    1
    -------------------------------------------------------------------------------
    2  ego  ■ --♦  static obstacle-- □ --> moving obstacle ----> reference path
    ===============================================================================
    """

    def __init__(self, key:PRNGKey, num_agents:int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(OvertakeInLowSpeed, self).__init__(key, num_agents)
        self._xrange = xrange
        self._yrange = yrange
        self._num_ref_points = num_ref_points
        self._lane_width = lane_width
        self._lane_centers = lane_centers

    @property
    def name(self):
        return 'overtake_scene_with_2_lanes_in_low_speed_for_c8'

    @property
    def num_ref_points(self) -> int:
        return self._num_ref_points

    @property
    def state_dim(self) -> int:
        return 8 # state: x y vx vy θ dθ/dt bw bh

    @property
    def num_lanes(self):
        assert self.lane_centers.shape[0] == 2, '本场景仅支持两车道配置！'
        return 2

    @property
    def lane_centers(self):
        return self._lane_centers

    @property
    def xrange(self):
        return self._xrange

    @property
    def yrange(self):
        return self._yrange

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        ref_y_key, ref_vx_key, agent_x_key, agent_y_key, agent_vx_key, sobst_x_key, mobst_x_key, mobst_vx_key \
            = jr.split(self.key, 8)
        num_lanes = self.num_lanes

        # 生成轨迹
        ref_y = jr.choice(ref_y_key, self.lane_centers, shape=())
        ref_vx = jr.uniform(ref_vx_key, shape=(), dtype=jnp.float32, minval=20, maxval=30)
        anS_goals = generate_horizontal_path_points(self.xrange, self.num_agents, self.num_ref_points, ref_y, ref_vx)

        # 生成初始agent，x,y,vx都可不一样，其它的都是0
        a_agent_x = jr.uniform(agent_x_key, shape=(self.num_agents,), dtype=jnp.float32, minval=self.xrange[0],
                               maxval=self.xrange[0]*2/3 + self.xrange[1]/3) # 前1/3的xrange里面选取
        a_agent_y = jr.choice(agent_y_key, self.lane_centers, shape=(self.num_agents,))
        a_agent_vx = jr.uniform(agent_vx_key, shape=(self.num_agents,), dtype=jnp.float32, minval=25, maxval=30)
        aSm3_other0 = jnp.zeros((self.num_agents, self.state_dim-3), dtype=jnp.float32)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0],
                                         axis=1)

        # 生成静态障碍物，x坐标位于中点和2/3点之间，y坐标和参考轨迹重合，航向角为0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32,
                             minval=self.xrange[0]/2 + self.xrange[1]/2,
                             maxval=self.xrange[0]/3 + self.xrange[1]*2/3)
        sobst_y = ref_y
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., 0., 0., 0., 0.])

        # 生成动态障碍物，x坐标位于静态障碍物之前5米到1/3整段长度之间，y坐标与参考轨迹重合，航向角为0，vx随机选择一个很小值
        mobst_x = sobst_x + jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32,
                             minval=5,
                             maxval=(self.xrange[1]-self.xrange[0])/3)
        mobst_y = ref_y
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=2, maxval=5)
        S_mobst_state = jnp.stack([mobst_x, mobst_y, mobst_vx, 0., 0., 0., 0., 0.])
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals


class HandMadeScene(LaneChangeANDOvertakeScene):
    """此场景用于实地测试时的场景设计，图例如下：
    ==================================================
    1
    --------------------------------------------------- // 这里是c8园区实际上的边界
    2
    --------------------------------------------------
    3  ego  ■ -----  □ --slow--> ♦  static -- reference path -->
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(HandMadeScene, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'lanechange_scene_with_edge_static_obstacle_and_middle_slow_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 3, '本场景仅支持三车道配置！'
        return 3

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs]:
        # 生成轨迹
        start_y = -3.
        terminal_vx = 30.
        Sm3_other0 = jnp.zeros((self.state_dim - 3,), dtype=jnp.float32)
        anS_goals = generate_horizontal_path_points(self.xrange, self.num_agents, self.num_ref_points, start_y,
                                                    terminal_vx)

        # 生成初始agent
        agent_x = -80.
        a_agent_x = jnp.repeat(jnp.array([agent_x]), self.num_agents)
        a_agent_y = jnp.repeat(jnp.array([start_y]), self.num_agents)
        a_agent_vx = jnp.repeat(jnp.array([25.]), self.num_agents)
        aSm3_other0 = jnp.repeat(Sm3_other0[None, :], self.num_agents, axis=0)
        aS_agent_state = jnp.concatenate([a_agent_x[:, None], a_agent_y[:, None], a_agent_vx[:, None], aSm3_other0],
                                         axis=1)

        # 生成静态障碍物
        sobst_x = -30.
        sobst_y = -3.
        sobst_theta = 0.
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., sobst_theta, 0., 0., 0.])

        # 生成动态障碍物
        mobst_y = jnp.array([-3.])
        mobst_vx = jnp.array([2.])
        mobst_x = jnp.array([30.])
        S_mobst_state = jnp.concatenate([mobst_x, mobst_y, mobst_vx, Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)

        return aS_agent_state, oS_obst_state, anS_goals

def gen_scene_randomly(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                       lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs]:
    choose_key, scene_key = jr.split(key, 2)
    scene_list = [LaneChangeMiddleStaticEdgeFastMoving(scene_key, num_agents, num_ref_points, xrange, yrange,
                                                       lane_width, lane_centers).make,
                 LaneChangeMiddleStaticEdgeSlowMoving(scene_key, num_agents, num_ref_points, xrange, yrange,
                                                      lane_width, lane_centers).make,
                 LaneChangeEdgeStaticMiddleFastMoving(scene_key, num_agents, num_ref_points, xrange, yrange,
                                                      lane_width, lane_centers).make,
                 LaneChangeEdgeStaticMiddleSlowMoving(scene_key, num_agents, num_ref_points, xrange, yrange,
                                                      lane_width, lane_centers).make,
                 OvertakeEdgeStaticMiddleFastMoving(scene_key, num_agents, num_ref_points, xrange, yrange,
                                                    lane_width, lane_centers).make,
                 OvertakeEdgeStaticMiddleSlowMoving(scene_key, num_agents, num_ref_points, xrange, yrange,
                                                    lane_width, lane_centers).make]
    # scene_list = [OvertakeInLowSpeed(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make]
    choose_id = jr.choice(choose_key, len(scene_list))
    aS_agent_state, oS_obst_state, anS_goals = jax.lax.switch(choose_id, scene_list)

    return aS_agent_state, oS_obst_state, anS_goals


def gen_handmade_scene_randomly(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                                lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs]:
    choose_key, scene_key = jr.split(key, 2)
    scene_list = [HandMadeScene(scene_key, num_agents, num_ref_points, xrange, yrange,
                                lane_width, lane_centers).make]
    choose_id = jr.choice(choose_key, len(scene_list))
    aS_agent_state, oS_obst_state, anS_goals = jax.lax.switch(choose_id, scene_list)

    return aS_agent_state, oS_obst_state, anS_goals