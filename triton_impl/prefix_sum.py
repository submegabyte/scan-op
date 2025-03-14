## naive prefix sum
## https://chatgpt.com/share/67d35b24-f160-800c-97ca-5cf0e7593350

import torch
import triton
import triton.language as tl

@triton.jit
def prefix_sum_kernel(X, Y, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    # Load elements
    x = tl.load(X + offset, mask=mask, other=0)

    # Perform warp-level scan (using sequential scan within a warp)
    for i in range(1, BLOCK_SIZE):
        x += tl.where(offset >= i, tl.load(X + offset - i, mask=mask, other=0), 0)

    # Store results
    tl.store(Y + offset, x, mask=mask)

def prefix_sum(X):
    N = X.numel()
    BLOCK_SIZE = 1024  # Adjust for GPU warp size
    Y = torch.empty_like(X)

    grid = (N // BLOCK_SIZE + 1,)
    prefix_sum_kernel[grid](X, Y, N, BLOCK_SIZE)

    return Y

# Example usage
X = torch.arange(10, dtype=torch.float32, device="cuda")
Y = prefix_sum(X)
print(Y)
