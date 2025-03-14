import jax
import jax.numpy as jnp

xs = jnp.array([1, 2, 3, 4, 5])
ys = jnp.array([4, 5, 8, 9, 11])

# # Prefix sum using addition
# result = jax.lax.associative_scan(jnp.add, xs)
# print(result)  # Output: [1, 3, 6, 10, 15]

# # Suffix sum using reverse scan
# result = jax.lax.associative_scan(jnp.add, xs, reverse=True)
# print(result)  # Output: [15, 14, 12, 9, 5]

# print(jnp.add(xs, ys))