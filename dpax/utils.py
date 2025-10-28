import jax
import jax.numpy as jnp

def create_2D_rect(length, width):
    #创建二维平面内的长方形，其边缘及内部由Ax<=b表示
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