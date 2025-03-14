## selective scan
## https://chatgpt.com/share/67d35b24-f160-800c-97ca-5cf0e7593350

import torch
import triton
import triton.language as tl

# input = A, B, X, C
# output = Y
# where H[i+1] = A[i] H[i] + B[i] X[i]
# and Y[i] = C[i] H[i]

@triton.jit
def selective_scan_kernel(A, B, C, X, Y, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    # Load inputs
    a = tl.load(A + offset, mask=mask, other=1.0)  # Default A = 1 for stability
    b = tl.load(B + offset, mask=mask, other=0.0)
    x = tl.load(X + offset, mask=mask, other=0.0)
    c = tl.load(C + offset, mask=mask, other=1.0)  # Default C = 1

    # Allocate space for H
    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Initialize H[0] to 0 (or a user-defined initial condition)
    if offset[0] == 0:
        h[0] = 0.0  # Change if you have a custom H[0] initialization
    
    # Sequential scan within the block
    for i in range(1, BLOCK_SIZE):
        if offset[i] < N:
            h[i] = a[i-1] * h[i-1] + b[i-1] * x[i-1]

    # Compute Y[i] = C[i] * H[i]
    y = c * h

    # Store results
    tl.store(Y + offset, y, mask=mask)

def selective_scan(A, B, C, X):
    N = A.numel()
    BLOCK_SIZE = 1024  # Adjust to match GPU properties
    Y = torch.zeros_like(A)  # Output array

    grid = (N // BLOCK_SIZE + 1,)
    selective_scan_kernel[grid](A, B, X, C, Y, N, BLOCK_SIZE)

    return Y

# Example usage
N = 10
A = torch.full((N,), 2.0, dtype=torch.float32, device="cuda")  # Example A = 2
B = torch.full((N,), 1.0, dtype=torch.float32, device="cuda")  # Example B = 1
X = torch.arange(N, dtype=torch.float32, device="cuda")  # X = [0,1,2,...]
C = torch.full((N,), 3.0, dtype=torch.float32, device="cuda")  # Example C = 3

Y = selective_scan(A, B, X, C)
print(Y)
