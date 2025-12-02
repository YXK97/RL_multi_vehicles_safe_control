import optax
import flax.linen as nn
import jax.numpy as jnp

from typing import Any, Callable, Literal, Sequence, Iterable, Generator, TypeVar
from jaxtyping import Array, Float
from flax import traverse_util


ActFn = Callable[[Array], Array]
PRGNKey = Float[Array, '2']
AnyFloat = Float[Array, '*']
Shape = tuple[int, ...]
InitFn = Callable[[PRGNKey, Shape, Any], Any]
HidSizes = Sequence[int]


_Elem = TypeVar("_Elem")


default_nn_init = nn.initializers.orthogonal


def scaled_init(initializer: nn.initializers.Initializer, scale: float) -> nn.initializers.Initializer:
    def scaled_init_inner(*args, **kwargs) -> AnyFloat:
        return scale * initializer(*args, **kwargs)

    return scaled_init_inner


ActLiteral = Literal["relu", "tanh", "elu", "swish", "silu", "gelu", "softplus"]


def get_act_from_str(act_str: ActLiteral) -> ActFn:
    act_dict: dict[Literal, ActFn] = dict(
        relu=nn.relu, tanh=nn.tanh, elu=nn.elu, swish=nn.swish, silu=nn.silu, gelu=nn.gelu, softplus=nn.softplus
    )
    return act_dict[act_str]


def signal_last_enumerate(it: Iterable[_Elem]) -> Generator[tuple[bool, int, _Elem], None, None]:
    iterable = iter(it)
    count = 0
    ret_var = next(iterable)
    for val in iterable:
        yield False, count, ret_var
        count += 1
        ret_var = val
    yield True, count, ret_var


def safe_get(arr, idx, fill_value=jnp.nan):
    return arr.at[idx].get(mode='fill', fill_value=fill_value)


def wd_mask(params):
    Path = tuple[str, ...]
    flat_params: dict[Path, jnp.ndarray] = traverse_util.flatten_dict(params)
    # Apply weight decay to all parameters except biases and LayerNorm scale and bias.
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)


def optim(learning_rate: float, wd: float, eps: float):
    opt = optax.adamw(learning_rate, eps=eps, weight_decay=wd, mask=wd_mask)
    opt = optax.apply_if_finite(opt, 1_000_000)
    return opt


def get_default_tx(
    lr: optax.Schedule | float, wd: optax.Schedule | float = 1e-4, eps: float = 1e-5
) -> optax.GradientTransformation:
    return optax.inject_hyperparams(optim)(learning_rate=lr, wd=wd, eps=eps)