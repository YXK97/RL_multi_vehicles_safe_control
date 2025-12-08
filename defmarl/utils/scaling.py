import jax
import jax.numpy as jnp

from .typing import Array, Pos2d, State
from .utils import calc_2d_rot_matrix

@jax.jit
def compute_intersections(O: Pos2d, V: Pos2d, A, b, fill: Pos2d, eps=1e-8) -> Array:
    """计算输入两点的直线与Ax=b各边的交点"""
    # 解析输入直线系数（向量化扩展为(k,)形状，适配A的q条边）
    a0 = V[1] - O[1]
    b0 = O[0] - V[0]
    c0 = V[0] * O[1] - O[0] * V[1]
    max_abs = jnp.array([jnp.abs(a0), jnp.abs(b0), jnp.abs(c0)]).max()
    a0 = a0 / max_abs
    b0 = b0 / max_abs
    c0 = c0 / max_abs
    a0 = jnp.full_like(A[:, 0], a0)  # (k,)
    b0 = jnp.full_like(A[:, 0], b0)  # (k,)
    c0 = jnp.full_like(A[:, 0], c0)  # (k,)
    # 解析A*x = b的q条边的系数（每条边：ai x + bi y - b_i' = 0）
    ai = A[:, 0]  # (k,)，每条边的x系数
    bi = A[:, 1]  # (k,)，每条边的y系数
    ci = -b  # (k,)，每条边的常数项（ai x + bi y + ci = 0 → ci = -b_i'）
    # 向量化计算行列式D（判断平行/相交）
    D = a0 * bi - ai * b0  # (k,)，批量计算k个行列式
    # 向量化计算交点分子（避免除法提前引入NaN）
    x_num = b0 * ci - bi * c0  # (k,)，x坐标分子
    y_num = ai * c0 - a0 * ci  # (k,)，y坐标分子
    # 处理交点：D≠0时计算真实交点，D≈0时设为inf（平行/重合）
    x = jnp.where(jnp.abs(D) > eps, x_num / D, fill[0])
    y = jnp.where(jnp.abs(D) > eps, y_num / D, fill[1])
    # 后续处理
    cand = jnp.stack([x, y], axis=1)
    cand = filter_ray_direction(cand, O, V, fill)
    cand = filter_in_bound(cand, A, b, fill)

    return cand

@jax.jit
def filter_ray_direction(k2_intersections: Array, O: Pos2d, V: Pos2d, fill: Pos2d) -> Array:
    """筛选与射线方向一致”的交点（射线：从O指向V）"""
    # 射线方向向量
    dir_vec = V - O  # (2,)
    # 交点相对于r1的向量：inter_vec = 交点 - r1
    inter_vec = k2_intersections - O  # (k, 2)
    # 点积≥0（同向）：dir_vec · inter_vec ≥ 0
    k_same_dir = jnp.sum(dir_vec * inter_vec, axis=1) >= -1e-8  # (k,)

    # 有效交点：同向，否则设为fill
    return jnp.where(k_same_dir[:, None], k2_intersections, fill)

@jax.jit
def filter_in_bound(k2_intersections: Array, A, b, fill: Pos2d) -> Array:
    """筛选满足Ax<=b的点"""
    def _single_in_bound(intersection: Pos2d, A, b) -> bool:
        return (jnp.dot(A, intersection)-b <= 1e-4).all()
    k_in_bound = jax.vmap(_single_in_bound, in_axes=(0, None, None))(k2_intersections, A, b)

    return jnp.where(k_in_bound[:, None], k2_intersections, fill)


@jax.jit
def scaling_calc(s1: State, s2: State) -> Array:
    """计算agent和agent/obst的scaling factor"""
    # state: x y vx vy θ dθdt bb_w bb_h a0 a1 a2 a3 a4 a5
    O1 = s1[:2]; O2 = s2[:2]
    # 计算 host 和 agent/obst 的顶点， host/agent/obst均为矩形
    Q1 = calc_2d_rot_matrix(s1[4])
    m_V = jnp.array([[s1[6] / 2, s1[7] / 2],
                    [s1[6] / 2, -s1[7] / 2],
                    [-s1[6] / 2, s1[7] / 2],
                    [-s1[6] / 2, -s1[7] / 2]])
    m_V = O1 + m_V @ Q1
    Q2 = calc_2d_rot_matrix(s2[4])
    n_P = jnp.array([[s2[6] / 2, s2[7] / 2],
                    [s2[6] / 2, -s2[7] / 2],
                    [-s2[6] / 2, s2[7] / 2],
                    [-s2[6] / 2, -s2[7] / 2]])
    n_P = O2 + n_P @ Q2

    # 计算S1和S2的A1 b1 A2 b2
    Ao = jnp.array([[ 1., 0.],  # x<=b/2
                    [-1., 0.],  # x>=-b/2
                    [ 0., 1.],  # y<=h/2
                    [ 0.,-1.]]) # y>=-h/2
    A1 = Ao @ Q1.T
    A2 = Ao @ Q2.T
    b1o = jnp.array([s1[6] / 2, s1[6] / 2, s1[7] / 2, s1[7] / 2])
    b2o = jnp.array([s2[6] / 2, s2[6] / 2, s2[7] / 2, s2[7] / 2])
    b1 = b1o + A1 @ O1
    b2 = b2o + A2 @ O2

    # host 向 自身顶点 发射射线
    mk2_intersections = jax.vmap(compute_intersections, in_axes=(None, 0, None, None, None))(
        O1, m_V, A2, b2, O1+1e8)
    mk_dist = jnp.linalg.norm(mk2_intersections - O1, axis=-1)
    mk_dist0 = jnp.linalg.norm(m_V - O1, axis=-1)[:, None].repeat(mk_dist.shape[1], axis=1)
    mk_scaling = mk_dist / mk_dist0
    # host 向 对方极点 发射射线
    nl2_intersections = jax.vmap(compute_intersections, in_axes=(None, 0, None, None, None))(
        O1, n_P, A1, b1, O1+1e-8)
    nl_dist = jnp.linalg.norm(nl2_intersections - O1, axis=-1)
    nl_dist0 = jnp.linalg.norm(n_P - O1, axis=-1)[:, None].repeat(nl_dist.shape[1], axis=1)
    nl_scaling = nl_dist0 / nl_dist
    scaling = jnp.array([mk_scaling.min(), nl_scaling.min()]).min()

    # 判断S1缩放中心是否在S2中
    O_in_S2 = jnp.where((A2@O1-b2 <= 1e-4).all(), -1., 1.)

    alpha = O_in_S2 * scaling
    return alpha

@jax.jit
def scaling_calc_bound(s: State, A: Array, b: Array) -> Array:
    """计算agent和bound的scaling factor，agent为有界多面凸集，bound为Ax<=b描述的单条直线分割的半空间，即A和b均为一行"""
    O = s[:2]
    # 计算host的顶点，host为矩形
    Q = calc_2d_rot_matrix(s[4])
    m_V = jnp.array([[ s[6]/2,  s[7]/2],
                     [ s[6]/2, -s[7]/2],
                     [-s[6]/2,  s[7]/2],
                     [-s[6]/2, -s[7]/2]])
    m_V = O + m_V @ Q

    # host向自身顶点发射射线
    mk2_intersections = jax.vmap(compute_intersections, in_axes=(None, 0, None, None, None))(
        O, m_V, A, b, O + 1e8)
    mk_dist = jnp.linalg.norm(mk2_intersections - O, axis=-1)
    mk_dist0 = jnp.linalg.norm(m_V - O, axis=-1)[:, None].repeat(mk_dist.shape[1], axis=1)
    scaling = (mk_dist / mk_dist0).min()

    # 判断S1缩放中心是否在bound中
    O_in_bound = jnp.where((A @ O - b <= 0).all(), -1., 1.)

    alpha = O_in_bound * scaling
    return alpha

