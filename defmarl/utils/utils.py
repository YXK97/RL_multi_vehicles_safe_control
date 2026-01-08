import pathlib
import jax.lax as lax
import einops as ei
import jax
import matplotlib.collections as mcollections
import numpy as np

from datetime import timedelta
from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar, Tuple, List, NamedTuple
from jax import numpy as jnp, tree_util as jtu
from jax._src.lib import xla_client as xc
from matplotlib.animation import FuncAnimation
from rich.progress import Progress, ProgressColumn
from rich.text import Text

from .typing import Array, Shape, BoolScalar, PathEff, AgentState, State


def merge01(x):
    return ei.rearrange(x, "n1 n2 ... -> (n1 n2) ...")


_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]

_PyTree = TypeVar("_PyTree")
_Arr = TypeVar("_Arr", np.ndarray, jnp.ndarray, bool)
_T = TypeVar("_T")
_U = TypeVar("_U")


def jax_vmap(fn: _Fn, in_axes: int | Sequence[Any] = 0, out_axes: Any = 0) -> _Fn:
    return jax.vmap(fn, in_axes, out_axes)


def concat_at_front(
        arr1: jnp.ndarray | np.ndarray, arr2: jnp.ndarray | np.ndarray, axis: int) -> jnp.ndarray | np.ndarray:
    """
    :param arr1: (nx, )
    :param arr2: (T, nx)
    :param axis: Which axis for arr2 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr2_shape = list(arr2.shape)
    del arr2_shape[axis]
    assert np.all(np.array(arr2_shape) == np.array(arr1.shape))

    if isinstance(arr1, np.ndarray):
        return np.concatenate([np.expand_dims(arr1, axis=axis), arr2], axis=axis)
    else:
        return jnp.concatenate([jnp.expand_dims(arr1, axis=axis), arr2], axis=axis)


def tree_concat_at_front(tree1: _PyTree, tree2: _PyTree, axis: int) -> _PyTree:
    def tree_concat_at_front_inner(arr1: jnp.ndarray, arr2: jnp.ndarray):
        return concat_at_front(arr1, arr2, axis=axis)

    return jtu.tree_map(tree_concat_at_front_inner, tree1, tree2)


def tree_index(tree: _PyTree, idx: int | Array) -> _PyTree:
    return jtu.tree_map(lambda x: x[idx], tree)

def tree_2nd_index(tree: _PyTree, idx: int | Array) -> _PyTree:
    return jtu.tree_map(lambda x: x[:, idx], tree)


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)


def np2jax(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(jnp.array, pytree)


def mask2index(mask: jnp.ndarray, n_true: int) -> jnp.ndarray:
    idx = lax.top_k(mask, n_true)[1]
    return idx


def jax_jit_np(
        fn: _Fn,
        static_argnums: int | Sequence[int] | None = None,
        static_argnames: str | Iterable[str] | None = None,
        donate_argnums: int | Sequence[int] = (),
        device: xc.Device = None,
        *args,
        **kwargs,
) -> _Fn:
    jit_fn = jax.jit(fn, static_argnums, static_argnames, donate_argnums, device, *args, **kwargs)

    def wrapper(*args, **kwargs) -> _R:
        return jax2np(jit_fn(*args, **kwargs))

    return wrapper


def chunk_vmap(fn: _Fn, chunks: int) -> _Fn:
    fn_jit_vmap = jax_jit_np(jax.vmap(fn))

    def wrapper(*args) -> _R:
        args = list(args)
        # 1: Get the batch size.
        batch_size = len(jtu.tree_leaves(args[0])[0])
        chunk_idxs = np.array_split(np.arange(batch_size), chunks)

        out = []
        for idxs in chunk_idxs:
            chunk_input = jtu.tree_map(lambda x: x[idxs], args)
            out.append(fn_jit_vmap(*chunk_input))
        
        # 2: Concatenate the output.
        out = tree_merge(out)
        return out

    return wrapper


class MutablePatchCollection(mcollections.PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self._paths = None
        self.patches = patches
        mcollections.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


class CustomTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        delta = timedelta(seconds=elapsed)
        delta = timedelta(seconds=delta.seconds, milliseconds=round(delta.microseconds // 1000))
        delta_str = str(delta)
        return Text(delta_str, style="progress.elapsed")


def save_anim(ani: FuncAnimation, path: pathlib.Path):
    pbar = Progress(*Progress.get_default_columns(), CustomTimeElapsedColumn())
    pbar.start()
    task = pbar.add_task("Animating", total=ani._save_count)

    def progress_callback(curr_frame: int, total_frames: int):
        pbar.update(task, advance=1)

    ani.save(path, progress_callback=progress_callback)
    pbar.stop()


def tree_merge(data: List[NamedTuple]):
    def body(*x):
        x = list(x)
        if isinstance(x[0], np.ndarray):
            return np.concatenate(x, axis=0)
        else:
            return jnp.concatenate(x, axis=0)
    out = jtu.tree_map(body, *data)
    return out


def tree_stack(trees: list):
    def tree_stack_inner(*arrs):
        arrs = list(arrs)
        if isinstance(arrs[0], np.ndarray):
            return np.stack(arrs, axis=0)
        return np.stack(arrs, axis=0)

    return jtu.tree_map(tree_stack_inner, *trees)


def as_shape(shape: int | Shape) -> Shape:
    if isinstance(shape, int):
        shape = (shape,)
    if not isinstance(shape, tuple):
        raise ValueError(f"Expected shape {shape} to be a tuple!")
    return shape


def get_or(maybe: _T | None, value: _U) -> _T | _U:
    return value if maybe is None else maybe


def assert_shape(arr: _Arr, shape: int | Shape, label: str | None = None) -> _Arr:
    shape = as_shape(shape)
    label = get_or(label, "array")
    if arr.shape != shape:
        raise AssertionError(f"Expected {label} of shape {shape}, but got shape {arr.shape} of type {type(arr)}!")
    return arr


def tree_where(cond: BoolScalar | bool, true_val: _PyTree, false_val: _PyTree) -> _PyTree:
    return jtu.tree_map(lambda x, y: jnp.where(cond, x, y), true_val, false_val)


@jax.jit
def calc_2d_rot_matrix(angle: float) -> Array:
    "计算二维平面的旋转矩阵，angle输入为degree: xO = Q·xb，这里的x0和xb均为列向量"
    angle = angle * jnp.pi / 180
    return jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                      [jnp.sin(angle), jnp.cos(angle)]]) # Q


def parse_jax_array(s: str) -> jnp.ndarray:
    """定义解析函数：将字符串转为jax.numpy数组"""
    s = s.replace(' ', '')  # 去除所有空格
    if ';' in s:
        rows = s.split(';')
        arr = [row.split(',') for row in rows]
    else:
        arr = s.split(',')
    return jnp.array(arr, dtype=jnp.float32)

@jax.jit
def calc_quintic_eff(starts: AgentState, terminals: AgentState) -> Tuple[PathEff, PathEff, PathEff]:
     """根据起点和终点求解五次多项式，输出原始参数、一阶导参数和二阶导参数"""
     zeros = jnp.zeros((starts.shape[0],), dtype=jnp.float32)
     # state: x y vx vy θ dθdt bw bh
     def A_b_create_and_solve(start, terminal) -> PathEff:
         x0 = start[0]
         x1 = terminal[0]
         A = jnp.array([[1, x0, x0**2,   x0**3,    x0**4,    x0**5],
                        [0,  1,  2*x0, 3*x0**2,  4*x0**3,  5*x0**4],
                        [0,  0,     2,    6*x0, 12*x0**2, 20*x0**3],
                        [1, x1, x1**2,   x1**3,    x1**4,    x1**5],
                        [0,  1,  2*x1, 3*x1**2,  4*x1**3,  5*x1**4],
                        [0,  0,     2,    6*x1, 12*x1**2, 20*x1**3],])
         y0 = start[1]
         y1 = terminal[1]
         t0 = start[4]*jnp.pi/180
         t1 = terminal[4]*jnp.pi/180
         b = jnp.array([y0, jnp.tan(t0), 0, y1, jnp.tan(t1), 0])
         coeff = jnp.linalg.solve(A, b)
         return coeff
     coeffs_f = jax.vmap(A_b_create_and_solve, in_axes=(0, 0))(starts, terminals)
     coeffs_df = jnp.stack([coeffs_f[:,1],2*coeffs_f[:,2],3*coeffs_f[:,3],4*coeffs_f[:,4],5*coeffs_f[:,5],zeros], axis=1)
     coeffs_ddf = jnp.stack([2*coeffs_f[:,2],6*coeffs_f[:,3],12*coeffs_f[:,4],20*coeffs_f[:,5],zeros,zeros], axis=1)
     return coeffs_f, coeffs_df, coeffs_ddf

@jax.jit
def calc_linear_eff(starts: AgentState, terminals: AgentState) -> Tuple[PathEff, PathEff, PathEff]:
     """根据起点和终点求解一次函数，输出原始参数、一阶导参数和二阶导参数"""
     zeros = jnp.zeros((starts.shape[0],), dtype=jnp.float32)
     # state: x y vx vy θ dθdt bw bh
     def A_b_create_and_solve(start, terminal) -> PathEff:
         x0 = start[0]
         x1 = terminal[0]
         A = jnp.array([[1, x0],
                        [1, x1]])
         y0 = start[1]
         y1 = terminal[1]
         b = jnp.array([y0, y1])
         coeff = jnp.linalg.solve(A, b)
         return coeff
     coeffs_f = jax.vmap(A_b_create_and_solve, in_axes=(0, 0))(starts, terminals)
     coeffs_df = jnp.stack([coeffs_f[:,1], zeros], axis=1)
     coeffs_ddf = jnp.stack([zeros, zeros], axis=1)
     return coeffs_f, coeffs_df, coeffs_ddf


def const_f(constant: Array) -> Callable:
    """构造常量函数"""
    def f(x):
        return constant
    return f

def linear_f(ai: Array) -> Callable:
    """根据ai构建y=a0+a1x的一次函数"""
    def f(x):
        return ai[:,0][:, None] + ai[:, 1][:, None]*x
    return f

def quintic_polynomial_f(ai: Array) -> Callable:
    """根据ai构建y=a0+a1x+a2x^2+a3x^3+a4x^4+a5x^5的五次函数"""
    def f(x):
        return (ai[:, 0][:, None] + ai[:, 1][:, None]*x + ai[:, 2][:, None]*x**2
                + ai[:, 3][:, None]*x**3 + ai[:, 4][:, None]*x**4 + ai[:, 5][:, None]*x**5)
    return f

def sin_f(A:Array, w:Array, T:Array, B:Array) -> Callable:
    """构建三角函数y=Asin(wx+T)+B"""
    def f(x):
        return A[:, None] * jnp.sin(w[:, None] * x + T[:, None]) + B[:, None]
    return f

def three_sec_f(f_l: Callable, f_m: Callable, f_h: Callable, x1: Array, x2: Array) -> Callable:
    """构造如下函数：
        { f_l(x), x <= x1
    y = { f_m(x), x1 < x <= x2
        { f_h(x), x > x2
    可处理向量化的x
    """
    def f(x):
        return jnp.where(x <= x1, f_l(x),
                         jnp.where(x > x2, f_h(x), f_m(x)))
    return f

def six_sec_f(f0: Callable, f1: Callable, f2: Callable, f3: Callable, f4: Callable, f5: Callable,
              x0: Array, x1: Array, x2: Array, x3: Array, x4: Array) -> Callable:
    """构造如下函数：
        { f0(x), x <= x0
        { f1(x), x0 < x <= x1
    y = { f2(x), x1 < x <= x2
        { f3(x), x2 < x <= x3
        { f4(x), x3 < x <= x4
        { f5(x), x > x4
    可处理向量化的x
    """
    def f(x):
        return jnp.where(x <= x0, f0(x),
                jnp.where(x <= x1, f1(x),
                 jnp.where(x <= x2, f2(x),
                  jnp.where(x <= x3, f3(x),
                   jnp.where(x <= x4, f4(x), f5(x))))))
    return f

@jax.jit
def find_closest_goal_indices(as_agent_states: AgentState, ans_all_goals: State) -> jnp.ndarray:
    """找到每个agent对应的最近goal的索引"""
    a2_agent_coords = as_agent_states[:, :2]
    an2_goal_coords = ans_all_goals[:, :, :2]
    a12_agent_coords = a2_agent_coords[:, None, :]
    an_sq_dists = jnp.sum((an2_goal_coords - a12_agent_coords) ** 2, axis=2)
    a_indices = jnp.argmin(an_sq_dists, axis=1)

    return a_indices

def gen_i_j_pairs(m: float, n: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """一组m个物体和一组n个物体间生成有序索引对i_pairs和j_pairs，其中i可以等于j"""
    i_grid, j_grid = jnp.meshgrid(jnp.arange(m), jnp.arange(n), indexing='ij')
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    return i_flat, j_flat

def gen_i_j_pairs_no_identical(m: float, n: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """一组m个物体和一组n个物体间生成有序索引对i_pairs和j_pairs，其中i不能等于j"""
    i_grid, j_grid = jnp.meshgrid(jnp.arange(m), jnp.arange(n), indexing='ij')
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    mask = i_flat != j_flat
    i_pairs = i_flat[mask]
    j_pairs = j_flat[mask]
    return i_pairs, j_pairs

@jax.jit
def normalize_angle(angles: jnp.ndarray) -> jnp.ndarray:
    """归一化角度到 [-180, 180]°，输入角度单位为°"""
    return (angles + 180) % 360 - 180


@jax.jit
def nthroot(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    复刻MATLAB的nthroot函数：计算实数的n次方根（返回实数，而非复数）
    参数：
        x: 底数（实数/实数数组）
        n: 根次数（正整数）
    返回：
        实数n次方根，兼容JAX向量化、JIT编译
    异常：
        1. n非正整数 → 报错
        2. n为偶数且x<0 → 报错
    """
    # 检查n是否为正整数
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"根次数n必须是正整数，当前n={n}")
    # 转换为jax数组（兼容标量/数组输入）
    x = jnp.asarray(x)
    # 偶数根：检查x是否非负
    if n % 2 == 0:
        if jnp.any(x < 0):
            raise ValueError(f"偶数根({n})的底数不能为负数，输入x包含负数：{x}")
        return jnp.pow(x, 1.0 / n)
    # 奇数根：处理负数底数（提取符号 + 绝对值开根 + 恢复符号）
    else:
        sign_x = jnp.sign(x)  # 提取符号（-1/0/1）
        abs_x = jnp.abs(x)  # 取绝对值
        root_abs = jnp.pow(abs_x, 1.0 / n)  # 绝对值开n次方
        return sign_x * root_abs