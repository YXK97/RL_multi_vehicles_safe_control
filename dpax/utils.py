import jax
import jax.numpy as jnp

from typing import Tuple
from .mrp import twoD_rotation_matrix
from .polytopes import polytope_proximity
from defmarl.utils.typing import SingleNode, Pos2d

def create_2D_rect(length, width):
    """创建二维平面内的长方形，其边缘及内部由Ax<=b表示"""
    A = jnp.array([
        [1., 0.],
        [0., 1.],
        [-1., 0.],
        [0., -1.]
    ])
    cs = jnp.array([
        [length / 2, 0.],
        [0., width / 2],
        [-length / 2, 0.],
        [0., -width / 2]
    ])
    b = jax.vmap(jnp.dot, in_axes=(0, 0))(A, cs)
    return A, b

def create_2D_hspace(A: jnp.ndarray, b: jnp.ndarray, r: Pos2d) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """二维半空间由Ax<=b定义，指定用于缩放的原点后，返回A(x-r)<=αb标准形式中的A,b,r"""
    return A, b-A.dot(r), r

def scaling_calc_between_recs(node1: SingleNode, node2: SingleNode) -> float:
    """通过两个节点(agent或obstacle)特征表征的矩形计算scaling factor"""
    A1, b1 = create_2D_rect(node1[4], node1[5])
    A2, b2 = create_2D_rect(node2[4], node2[5])
    r1 = node1[:2]
    r2 = node2[:2]
    Q1 = twoD_rotation_matrix(node1[3])
    Q2 = twoD_rotation_matrix(node2[3])
    alpha = polytope_proximity(A1, b1, r1, Q1, A2, b2, r2, Q2)
    return alpha

def scaling_calc_between_rec_and_hspace(node: SingleNode, A: jnp.ndarray, b: jnp.ndarray, r: Pos2d) -> float:
    """通过一个节点(agent或obstacle)特征表征的矩形和Ax<=b表征的半空间以及用于缩放的原点计算scaling factor"""
    A1, b1 = create_2D_rect(node[4], node[5])
    r1 = node[:2]
    A2, b2, r2 = create_2D_hspace(A, b, r)
    Q1 = twoD_rotation_matrix(node[3])
    Q2 = jnp.eye(Q1.shape[0])
    alpha = polytope_proximity(A1, b1, r1, Q1, A2, b2, r2, Q2)
    return alpha