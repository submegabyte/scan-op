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

A = jnp.arange(1, 10)
BX = jnp.arange(1, 10)

def state_kernel(s, c):
    sx, sy = s[0], s[1]
    cx, cy = c[0], c[1]

    sx = cx * sx
    sy = cx * sy + cy

    return sx, sy

result = jax.lax.associative_scan(state_kernel, (A, BX))
print(result)