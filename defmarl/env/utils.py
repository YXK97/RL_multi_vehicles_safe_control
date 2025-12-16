import numpy as np
import jax.numpy as jnp
import functools as ft
import jax
import jax.random as jr

from typing import Tuple
from jax.lax import while_loop

from ..utils.typing import Array, Radius, BoolScalar, Pos, PRNGKey, AgentState, PathRefs
from ..utils.utils import calc_linear_eff, calc_quintic_eff, const_f, linear_f, quintic_polynomial_f, sin_f, \
    three_sec_f, six_sec_f, calc_2d_rot_matrix
from .obstacle import Obstacle


def inside_obstacles(points: Pos, obstacles: Obstacle = None, r: Radius = 0.) -> BoolScalar:
    """
    points: (n, n_dim) or (n_dim, )
    obstacles: tree_stacked obstacles.

    Returns: (n, ) or (,). True if in collision, false otherwise.
    """
    if obstacles is None:
        if points.ndim == 1:
            return jnp.zeros((), dtype=bool)
        return jnp.zeros(points.shape[0], dtype=bool)

    # one point inside one obstacle
    def inside(point: Pos, obstacle: Obstacle):
        return obstacle.inside(point, r)

    # one point inside any obstacle
    def inside_any(point: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(inside, point))(obstacle).max()

    # any point inside any obstacle
    if points.ndim == 1:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros((), dtype=bool)
        is_in = inside_any(points, obstacles)
    else:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros(points.shape[0], dtype=bool)
        is_in = jax.vmap(ft.partial(inside_any, obstacle=obstacles))(points)

    return is_in


def get_node_goal_rng(
        key: PRNGKey,
        side_length: float | Array,
        dim: int,
        n: int,
        min_dist: float,
        obstacles: Obstacle = None,
        side_length_y: float = None,
        max_travel: float = None
) -> Tuple[Pos, Pos]:
    max_iter = 1024  # maximum number of iterations to find a valid initial state/goal
    states = jnp.zeros((n, dim))
    goals = jnp.zeros((n, dim))
    side_length_y = side_length if side_length_y is None else side_length_y

    def get_node(reset_input: Tuple[int, Array, Array, Array]):  # i_iter, key, node, all nodes
        i_iter, this_key, _, all_nodes = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        return i_iter, this_key, jr.uniform(use_key, (dim,),
                                            minval=0, maxval=jnp.array([side_length, side_length_y])), all_nodes

    def non_valid_node(reset_input: Tuple[int, Array, Array, Array]):  # i_iter, key, node, all nodes
        i_iter, _, node, all_nodes = reset_input
        dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
        collide = dist_min <= min_dist
        inside = inside_obstacles(node, obstacles, r=min_dist / 2)
        valid = ~(collide | inside) | (i_iter >= max_iter)
        return ~valid

    def get_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
        # i_iter, key, goal_candidate, agent_start_pos, all_goals
        i_iter, this_key, _, agent, all_goals = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        if max_travel is None:
            return (i_iter, this_key,
                    jr.uniform(use_key, (dim,), minval=0, maxval=jnp.array([side_length, side_length_y])),
                    agent, all_goals)
        else:
            return i_iter, this_key, jr.uniform(
                use_key, (dim,), minval=-max_travel, maxval=max_travel) + agent, agent, all_goals

    def non_valid_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
        # i_iter, key, goal_candidate, agent_start_pos, all_goals
        i_iter, _, goal, agent, all_goals = reset_input
        dist_min = jnp.linalg.norm(all_goals - goal, axis=1).min()
        collide = dist_min <= min_dist
        inside = inside_obstacles(goal, obstacles, r=min_dist / 2)
        outside = jnp.any(goal < 0) | jnp.any(goal > side_length)
        if max_travel is None:
            too_long = np.array(False, dtype=bool)
        else:
            too_long = jnp.linalg.norm(goal - agent) > max_travel
        valid = (~collide & ~inside & ~outside & ~too_long) | (i_iter >= max_iter)
        out = ~valid
        assert out.shape == tuple() and out.dtype == jnp.bool_
        return out

    def reset_body(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        agent_key, goal_key, this_key = jr.split(this_key, 3)
        agent_candidate = jr.uniform(agent_key, (dim,), minval=0, maxval=jnp.array([side_length, side_length_y]))
        n_iter_agent, _, agent_candidate, _ = while_loop(
            cond_fun=non_valid_node, body_fun=get_node,
            init_val=(0, agent_key, agent_candidate, all_states)
        )
        all_states = all_states.at[agent_id].set(agent_candidate)

        if max_travel is None:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=0, maxval=jnp.array([side_length, side_length_y]))
        else:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=0, maxval=max_travel) + agent_candidate

        n_iter_goal, _, goal_candidate, _, _ = while_loop(
            cond_fun=non_valid_goal, body_fun=get_goal,
            init_val=(0, goal_key, goal_candidate, agent_candidate, all_goals)
        )
        all_goals = all_goals.at[agent_id].set(goal_candidate)
        agent_id += 1

        # if no solution is found, start over
        agent_id = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * agent_id
        all_states = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_states
        all_goals = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_goals

        return agent_id, this_key, all_states, all_goals

    def reset_not_terminate(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        return agent_id < n

    _, _, states, goals = while_loop(
        cond_fun=reset_not_terminate, body_fun=reset_body, init_val=(0, key, states, goals))

    return states, goals

def process_lane_centers(y_state_range: Array, lane_width: float) -> Array:
    """根据输入的y坐标范围和车道宽度，解析所有车道中心线的位置并整合进一个数组之中"""
    yh = y_state_range[1]
    yl = y_state_range[0]
    n = jnp.floor((yh-yl)/lane_width).astype(int)
    i = jnp.arange(start=1, stop=n+1, step=1, dtype=int)
    c_ycs = yh - lane_width*(i-1/2)
    return c_ycs

def process_lane_marks(y_state_range: Array, lane_width: float) -> Tuple[Array, Array|None]:
    """根据输入的y坐标范围和车道宽度，解析所有车道边界的位置并整合进两个数组之中，第一个输出为道路边界，第二个输出为车道线（虚线）"""
    yh = y_state_range[1]
    yl = y_state_range[0]
    n = jnp.floor((yh - yl) / lane_width).astype(int)
    assert n>=1
    i = jnp.arange(start=1, stop=n+1, step=1, dtype=int)
    scatters = yh - lane_width*i
    if scatters[-1] == yl:
        scatters = scatters[:-1]
    bolds = y_state_range
    if scatters.shape[0] == 0:
        return bolds, None
    else:
        return bolds, scatters

def generate_goals(key:Array, centers: Array, agents: AgentState, terminals: AgentState, n_ref_pts: int, ref_pts_interval: float = 0.1) -> PathRefs:
    """根据输入生成轨迹参考点，agent初始状态和终点状态确定后（两者x距离至少为200m），随机从分段五次多项式曲线、折线、sin曲线中选择轨迹，轨迹初始点
    不一定和agent初始点重合（x坐标重合，y坐标不一定，纵向速度均由terminal确定，横向速度均为0，航向角和航向角速度均为0）

    分段五次多项式曲线：主要的训练用曲线，选取概率为50%，从起点至终点随机选择4个路径点，作为分段参考
    折线：选取概率为25%，路径点与分段五次多项式曲线的选择方法相同
    sin曲线：选取概率25%，函数为y=Asin(ωx+T)+B，其中需要确保max(|y|)<=4.5"""
    # key分配
    curve_choose_key, way_points_key, sin_key = jr.split(key, 3)

    # 参数提取
    num_agents = agents.shape[0]
    a_agents_x = agents[:, 0]
    a_terminals_x = terminals[:, 0]
    a_terminals_y = terminals[:, 1]
    a_terminals_v = terminals[:, 2]

    # 随机获取路径点
    L_key, x0_key, x123_key, y0123_key = jr.split(way_points_key, 4)
    a_L = jr.uniform(L_key, shape=(num_agents,), dtype=jnp.float32,
                     minval=200*jnp.ones((num_agents,)), maxval=(a_terminals_x-a_agents_x))
    a_x0 = jr.uniform(x0_key, shape=(num_agents,), dtype=jnp.float32,
                    minval=a_agents_x, maxval=(a_terminals_x-a_L))
    a_x4 = a_x0 + a_L
    i = jnp.repeat(jnp.arange(start=1, stop=4, step=1, dtype=int)[None,:], num_agents, axis=0)
    a3_x123 = jr.uniform(x123_key, shape=(num_agents, 3), dtype=jnp.float32,
                         minval=(a_L[:,None]*i/4-5)+a_x0[:,None], maxval=(a_L[:,None]*i/4+5)+a_x0[:,None])
    a4_y0123 = jr.choice(y0123_key, centers, shape=(num_agents, 4))
    a5_y01234 = jnp.concatenate([a4_y0123, a_terminals_y[:, None]], axis=1)
    a5_x01234 = jnp.concatenate([a_x0[:, None], a3_x123, a_x4[:, None]], axis=1)
    a52_pos01234 = jnp.stack([a5_x01234, a5_y01234], axis=2)
    a42_pos0123 = a52_pos01234[:, :-1, :]
    a4s_start_states = jnp.concatenate([a42_pos0123,
                                        jnp.repeat(a_terminals_v[:,None,None],4,axis=1),
                                        jnp.zeros((num_agents,4,5))], axis=2)
    assert a4s_start_states.shape[-1] == agents.shape[-1]
    a42_pos1234 = a52_pos01234[:, 1:, :]
    a4s_end_states = jnp.concatenate([a42_pos1234,
                                        jnp.repeat(a_terminals_v[:, None, None], 4, axis=1),
                                        jnp.zeros((num_agents, 4, 5))], axis=2)
    assert a4s_end_states.shape[-1] == agents.shape[-1]

    # 生成分段五次多项式
    a46_patheffs_f, a46_patheffs_df, a46_patheffs_ddf = jax.vmap(calc_quintic_eff, in_axes=(1, 1), out_axes=(1, 1, 1))(
        a4s_start_states, a4s_end_states)
    quintic_1_f, quintic_2_f, quintic_3_f, quintic_4_f = quintic_polynomial_f(a46_patheffs_f[:,0,:]), \
        quintic_polynomial_f(a46_patheffs_f[:, 1, :]), quintic_polynomial_f(a46_patheffs_f[:, 2, :]), \
        quintic_polynomial_f(a46_patheffs_f[:, 3, :])
    quintic_1_df, quintic_2_df, quintic_3_df, quintic_4_df = quintic_polynomial_f(a46_patheffs_df[:, 0, :]), \
        quintic_polynomial_f(a46_patheffs_df[:, 1, :]), quintic_polynomial_f(a46_patheffs_df[:, 2, :]), \
        quintic_polynomial_f(a46_patheffs_df[:, 3, :])
    quintic_1_ddf, quintic_2_ddf, quintic_3_ddf, quintic_4_ddf = quintic_polynomial_f(a46_patheffs_ddf[:, 0, :]), \
        quintic_polynomial_f(a46_patheffs_ddf[:, 1, :]), quintic_polynomial_f(a46_patheffs_ddf[:, 2, :]), \
        quintic_polynomial_f(a46_patheffs_ddf[:, 3, :])

    # 生成分段折线
    a42_patheffs_f, a42_patheffs_df, a42_patheffs_ddf = jax.vmap(calc_linear_eff, in_axes=(1, 1), out_axes=(1, 1, 1))(
        a4s_start_states, a4s_end_states)
    linear_1_f, linear_2_f, linear_3_f, linear_4_f = linear_f(a42_patheffs_f[:,0,:]), \
        linear_f(a42_patheffs_f[:,1,:]), linear_f(a42_patheffs_f[:,2,:]), linear_f(a42_patheffs_f[:,3,:])
    linear_1_df, linear_2_df, linear_3_df, linear_4_df = linear_f(a42_patheffs_df[:,0,:]), \
        linear_f(a42_patheffs_df[:,1,:]), linear_f(a42_patheffs_df[:,2,:]), linear_f(a42_patheffs_df[:,3,:])
    linear_1_ddf, linear_2_ddf, linear_3_ddf, linear_4_ddf = linear_f(a42_patheffs_ddf[:,0,:]), \
        linear_f(a42_patheffs_ddf[:,1,:]), linear_f(a42_patheffs_ddf[:,2,:]), linear_f(a42_patheffs_ddf[:,3,:])

    # 构建三个值的常数函数
    zeros = jnp.zeros((agents.shape[0], 1), dtype=jnp.float32)
    const_f_y0 = const_f(a5_y01234[:, 0][:, None])
    const_f_terminals_y = const_f(terminals[:, 1][:, None])
    const_f_zeros = const_f(zeros)

    # 构建中间为五次多项式的分段函数
    poly_sec_f = six_sec_f(const_f_y0, quintic_1_f, quintic_2_f, quintic_3_f, quintic_4_f, const_f_terminals_y,
        a52_pos01234[:,0,0][:,None], a52_pos01234[:,1,0][:,None], a52_pos01234[:,2,0][:,None],
            a52_pos01234[:,3,0][:,None], a52_pos01234[:,4,0][:,None])
    poly_sec_df = six_sec_f(const_f_zeros, quintic_1_df, quintic_2_df, quintic_3_df, quintic_4_df, const_f_zeros,
        a52_pos01234[:,0,0][:,None], a52_pos01234[:,1,0][:,None], a52_pos01234[:,2,0][:,None],
            a52_pos01234[:,3,0][:,None], a52_pos01234[:,4,0][:,None])
    poly_sec_ddf = six_sec_f(const_f_zeros, quintic_1_ddf, quintic_2_ddf, quintic_3_ddf, quintic_4_ddf, const_f_zeros,
        a52_pos01234[:,0,0][:,None], a52_pos01234[:,1,0][:,None], a52_pos01234[:,2,0][:,None],
            a52_pos01234[:,3,0][:,None], a52_pos01234[:,4,0][:,None])

    # 构建中间为折线的分段函数
    lin_sec_f = six_sec_f(const_f_y0, linear_1_f, linear_2_f, linear_3_f, linear_4_f, const_f_terminals_y,
        a52_pos01234[:,0,0][:,None], a52_pos01234[:,1,0][:,None], a52_pos01234[:,2,0][:,None],
            a52_pos01234[:,3,0][:,None], a52_pos01234[:,4,0][:,None])
    lin_sec_df = six_sec_f(const_f_zeros, linear_1_df, linear_2_df, linear_3_df, linear_4_df, const_f_zeros,
        a52_pos01234[:,0,0][:,None], a52_pos01234[:,1,0][:,None], a52_pos01234[:,2,0][:,None],
            a52_pos01234[:,3,0][:,None], a52_pos01234[:,4,0][:,None])
    lin_sec_ddf = six_sec_f(const_f_zeros, linear_1_ddf, linear_2_ddf, linear_3_ddf, linear_4_ddf, const_f_zeros,
        a52_pos01234[:,0,0][:,None], a52_pos01234[:,1,0][:,None], a52_pos01234[:,2,0][:,None],
            a52_pos01234[:,3,0][:,None], a52_pos01234[:,4,0][:,None])

    # 构建sin曲线参数
    A_key, w_key, T_key, B_key, = jr.split(sin_key, 4)
    a_As = jr.uniform(A_key, shape=(num_agents,), dtype=jnp.float32,
                      minval=jnp.zeros((num_agents,)), maxval=centers[0]*jnp.ones((num_agents,)))
    a_Bs = jr.uniform(B_key, shape=(num_agents,), dtype=jnp.float32,
                      minval=jnp.zeros((num_agents,)), maxval=centers[0]-a_As)
    a_ws = jr.uniform(w_key, shape=(num_agents,), dtype=jnp.float32,
                      minval=jnp.ones((num_agents,))*jnp.pi/160,
                      maxval=jnp.ones((num_agents,))*jnp.pi/80)
    a_Ts = jr.uniform(T_key, shape=(num_agents,), dtype=jnp.float32,
                      minval=jnp.zeros((num_agents,)), maxval=2*jnp.pi*jnp.ones((num_agents,)))
    sin_curve_f = sin_f(a_As, a_ws, a_Ts, a_Bs)
    sin_curve_df = sin_f(a_As*a_ws, a_ws, a_Ts+2/jnp.pi, jnp.zeros((num_agents,)))
    sin_curve_ddf = sin_f(-a_As*a_ws**2, a_ws, a_Ts, jnp.zeros((num_agents,)))

    # 构建路径点
    a_curve_choose = jr.choice(curve_choose_key, jnp.array([0, 1, 2, 3]), shape=(num_agents, 1)) # 路径选择变量
    an_xs = jnp.linspace(start=agents[:,0], stop=agents[:,0]+(n_ref_pts+1)*ref_pts_interval, num=n_ref_pts, dtype=jnp.float32).T
    an_ys = jnp.where(a_curve_choose <= 1, poly_sec_f(an_xs),
                       jnp.where(a_curve_choose > 2, sin_curve_f(an_xs), lin_sec_f(an_xs)))
    an_dys = jnp.where(a_curve_choose <= 1, poly_sec_df(an_xs),
                      jnp.where(a_curve_choose > 2, sin_curve_df(an_xs), lin_sec_df(an_xs)))
    an_ddys = jnp.where(a_curve_choose <= 1, poly_sec_ddf(an_xs),
                      jnp.where(a_curve_choose > 2, sin_curve_ddf(an_xs), lin_sec_ddf(an_xs)))

    an_thetas_rad = jnp.arctan(an_dys)
    an_thetas_deg = an_thetas_rad * 180/jnp.pi

    # state: x y vx vy θ dθdt bw bh
    an_vs_kmph = jnp.repeat(terminals[:, 2][:, None], an_thetas_rad.shape[1], axis=1)
    an_vxs_kmph = an_vs_kmph * jnp.cos(an_thetas_rad)
    an_vys_kmph = an_vs_kmph * jnp.sin(an_thetas_rad)

    an_dthetas_radps = an_ddys * an_vxs_kmph/3.6 / (1+an_dys**2)
    an_dthetas_degps = an_dthetas_radps * 180/jnp.pi

    an_zeros = jnp.zeros_like(an_xs)

    ans_goals = jnp.stack([an_xs, an_ys, an_vxs_kmph, an_vys_kmph, an_thetas_deg, an_dthetas_degps, an_zeros, an_zeros], axis=2)
    return ans_goals


def generate_horizontal_goals(agents: AgentState, terminals: AgentState, n_ref_pts: int, ref_pts_interval: float = 0.1) -> PathRefs:
    """根据agents起点和terminals终点生成y=0的参考点"""
    # 参数提取
    state_dim = agents.shape[1]
    num_agents = agents.shape[0]
    a_agents_x = agents[:, 0]
    a_terminals_vx = terminals[:, 2]
    # state: x y vx vy θ dθdt bw bh
    an_xs = jnp.linspace(start=a_agents_x, stop=a_agents_x+(n_ref_pts+1)*ref_pts_interval, num=n_ref_pts, dtype=jnp.float32).T
    num_points = an_xs.shape[1]
    ansm1_other_0s = jnp.zeros((num_agents, num_points, state_dim-1), dtype=jnp.float32)
    ansm1_other = ansm1_other_0s.at[:, :, 1].set(jnp.repeat(a_terminals_vx[:, None], num_points, axis=1))

    ans_goals = jnp.concatenate([an_xs[:, :, None], ansm1_other], axis=2)
    return ans_goals


def generate_lanechange_goals(way_points_key:Array, centers: Array, agents: AgentState, terminals: AgentState, n_ref_pts: int, ref_pts_interval: float = 0.1) -> PathRefs:
    """根据输入生成轨迹参考点，agent初始状态和终点状态确定后（两者x距离至少为200m），使用分段五次多项式生成变道一次的参考轨迹，轨迹初始点
    不一定和agent初始点重合（x坐标重合，y坐标不一定，纵向速度均由terminal确定，横向速度均为0，航向角和航向角速度均为0）"""
    # 参数提取
    num_agents = agents.shape[0]
    a_agents_x = agents[:, 0]
    a_terminals_x = terminals[:, 0]
    a_terminals_y = terminals[:, 1]
    a_terminals_v = terminals[:, 2]

    # 随机获取路径点
    L_key, xstart_key, ystart_key = jr.split(way_points_key, 3)
    a_L = jr.uniform(L_key, shape=(num_agents,), dtype=jnp.float32,
                     minval=100*jnp.ones((num_agents,)), maxval=(a_terminals_x-a_agents_x))
    a_xstart = jr.uniform(xstart_key, shape=(num_agents,), dtype=jnp.float32,
                          minval=a_agents_x, maxval=(a_terminals_x-a_L))
    a_xend = a_xstart + a_L
    a_ystart = jr.choice(ystart_key, centers, shape=(num_agents,))
    a2_ystartend = jnp.stack([a_ystart, a_terminals_y], axis=1)
    a2_xstartend = jnp.stack([a_xstart, a_xend], axis=1)
    a22_posstartend = jnp.stack([a2_xstartend, a2_ystartend], axis=2)
    a2_posstart = a22_posstartend[:, 0, :]
    as_start_state = jnp.concatenate([a2_posstart, a_terminals_v[:, None], jnp.zeros((num_agents,5))], axis=1)
    assert as_start_state.shape[-1] == agents.shape[-1]
    a2_posend = a22_posstartend[:, 1, :]
    as_end_state = jnp.concatenate([a2_posend, a_terminals_v[:, None], jnp.zeros((num_agents,5))], axis=1)
    assert as_end_state.shape[-1] == agents.shape[-1]

    # 生成中间的五次多项式
    a6_patheffs_f, a6_patheffs_df, a6_patheffs_ddf = calc_quintic_eff(as_start_state, as_end_state)
    quintic_f = quintic_polynomial_f(a6_patheffs_f)
    quintic_df = quintic_polynomial_f(a6_patheffs_df)
    quintic_ddf = quintic_polynomial_f(a6_patheffs_ddf)

    # 构建三个值的常数函数
    zeros = jnp.zeros((agents.shape[0], 1), dtype=jnp.float32)
    const_f_ystart = const_f(a2_ystartend[:, 0][:, None])
    const_f_terminals_y = const_f(terminals[:, 1][:, None])
    const_f_zeros = const_f(zeros)

    # 构建中间为五次多项式的分段函数
    poly_sec_f = three_sec_f(const_f_ystart, quintic_f, const_f_terminals_y,
                             a22_posstartend[:,0,0][:,None], a22_posstartend[:,1,0][:,None])
    poly_sec_df = three_sec_f(const_f_zeros, quintic_df, const_f_zeros,
                              a22_posstartend[:, 0, 0][:, None], a22_posstartend[:, 1, 0][:, None])
    poly_sec_ddf = three_sec_f(const_f_zeros, quintic_ddf, const_f_zeros,
                               a22_posstartend[:, 0, 0][:, None], a22_posstartend[:, 1, 0][:, None])

    # 构建路径点
    an_xs = jnp.linspace(start=agents[:,0], stop=agents[:,0]+(n_ref_pts+1)*ref_pts_interval, num=n_ref_pts, dtype=jnp.float32).T
    an_ys = poly_sec_f(an_xs)
    an_dys = poly_sec_df(an_xs)
    an_ddys =  poly_sec_ddf(an_xs)

    an_thetas_rad = jnp.arctan(an_dys)
    an_thetas_deg = an_thetas_rad * 180/jnp.pi

    # state: x y vx vy θ dθdt bw bh
    an_vs_kmph = jnp.repeat(terminals[:, 2][:, None], an_thetas_rad.shape[1], axis=1)
    an_vxs_kmph = an_vs_kmph * jnp.cos(an_thetas_rad)
    an_vys_kmph = an_vs_kmph * jnp.sin(an_thetas_rad)

    an_dthetas_radps = an_ddys * an_vxs_kmph/3.6 / (1+an_dys**2)
    an_dthetas_degps = an_dthetas_radps * 180/jnp.pi

    an_zeros = jnp.zeros_like(an_xs)

    ans_goals = jnp.stack([an_xs, an_ys, an_vxs_kmph, an_vys_kmph, an_thetas_deg, an_dthetas_degps, an_zeros, an_zeros], axis=2)
    return ans_goals


@jax.jit
def relative_state(ego_state: AgentState, target_state: AgentState) -> AgentState:
    """计算target在ego连体基下的状态量"""
    convert_vec = jnp.array([1, 1, 3.6, 3.6, 180/jnp.pi, 180/jnp.pi, 1, 1]) # eg. km/h / convert_vec -> m/s
    ego_state_metric = ego_state / convert_vec
    target_state_metric = target_state / convert_vec

    # 参数提取
    ego_pos_m, target_pos_m = ego_state_metric[:2], target_state_metric[:2]
    ego_v_mps, target_v_mps = ego_state_metric[2:4], target_state_metric[2:4]
    ego_theta_rad, target_theta_rad = ego_state_metric[4], target_state_metric[4]
    ego_omega_radps, target_omega_radps = ego_state_metric[5], target_state_metric[5]
    ego_b_m, target_b_m = ego_state_metric[6:], target_state_metric[6:]
    Q = calc_2d_rot_matrix(ego_theta_rad * 180/jnp.pi)

    # 相对角速度与相对角加速度
    rel_theta_rad = target_theta_rad - ego_theta_rad
    rel_omega_radps = target_omega_radps - ego_omega_radps

    # 相对坐标
    rel_pos_m = Q.T @ (target_pos_m - ego_pos_m)
    rel_x_m = rel_pos_m[0]
    rel_y_m = rel_pos_m[1]

    # 相对方向角与距离
    rel_angle_rad = jnp.arctan(rel_y_m / (rel_x_m + 1e-4))
    rel_dist_m = jnp.linalg.norm(rel_pos_m)

    # 相对速度
    rel_v_mps = Q.T @ (target_v_mps - ego_v_mps +
                       jnp.array([ego_omega_radps * rel_dist_m * jnp.sin(rel_angle_rad + ego_theta_rad),
                                 -ego_omega_radps * rel_dist_m * jnp.cos(rel_angle_rad + ego_theta_rad)]))

    # 相对包围盒参数
    rel_b_m = target_b_m - ego_b_m

    # 整合与单位转换
    rel_state_metric = jnp.concatenate([rel_pos_m, rel_v_mps, jnp.array([rel_theta_rad]), jnp.array([rel_omega_radps]),
                                        rel_b_m])
    rel_state = rel_state_metric * convert_vec

    assert rel_state.shape == ego_state.shape == target_state.shape
    return rel_state