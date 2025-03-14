import torch
import cuda_blelloch_scan  # Import the compiled CUDA extension

# Input data
sx = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
sy = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float32, device="cuda")
cx = torch.tensor([1, 2, 1, 0.5, 1], dtype=torch.float32, device="cuda")
cy = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device="cuda")

# Run the Blelloch associative scan
Sx, Sy = cuda_blelloch_scan.blelloch_associative_scan(sx, sy, cx, cy)

print("Input Sx:", sx.cpu().numpy())
print("Input Sy:", sy.cpu().numpy())
print("Input Cx:", cx.cpu().numpy())
print("Input Cy:", cy.cpu().numpy())

print("Output Sx:", Sx.cpu().numpy())
print("Output Sy:", Sy.cpu().numpy())
