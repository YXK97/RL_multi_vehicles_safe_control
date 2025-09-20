import jax.numpy as jnp

dist = jnp.array([[1.1, 0.5, 1.7],
                  [0.5, 1.1, 0.3],
                  [1.7, 0.3, 1.1]])

agent_agent_mask = jnp.less(dist, 1.)

print(agent_agent_mask)