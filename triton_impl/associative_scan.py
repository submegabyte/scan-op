## associative scan
## https://chatgpt.com/share/67d35b24-f160-800c-97ca-5cf0e7593350

import torch
import triton
import triton.language as tl

# input = A, B, X
# output = H
# where H[i+1] = A[i] H[i] + B[i] X[i]

@triton.jit
def associative_scan_kernel(A, B, X, H, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    # Load inputs
    a = tl.load(A + offset, mask=mask, other=1.0)  # Default A = 1 for stability
    b = tl.load(B + offset, mask=mask, other=0.0)
    x = tl.load(X + offset, mask=mask, other=0.0)
    
    # Allocate space for H
    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Initialize H[0] to 0 (or user-defined initial condition)
    if offset[0] == 0:
        h[0] = 0.0  # Change if you have a custom H[0] initialization
    
    # Sequential scan within the block
    for i in range(1, BLOCK_SIZE):
        if offset[i] < N:
            h[i] = a[i-1] * h[i-1] + b[i-1] * x[i-1]

    # Store results
    tl.store(H + offset, h, mask=mask)

def associative_scan(A, B, X):
    N = A.numel()
    BLOCK_SIZE = 1024  # Adjust to match GPU properties
    H = torch.zeros_like(A)  # Output array

    grid = (N // BLOCK_SIZE + 1,)
    associative_scan_kernel[grid](A, B, X, H, N, BLOCK_SIZE)

    return H

# Example usage
N = 10
A = torch.full((N,), 2.0, dtype=torch.float32, device="cuda")  # Example A = 2
B = torch.full((N,), 1.0, dtype=torch.float32, device="cuda")  # Example B = 1
X = torch.arange(N, dtype=torch.float32, device="cuda")  # X = [0,1,2,...]

H = associative_scan(A, B, X)
print(H)
