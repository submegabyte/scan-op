import torch
import cuda_scan  # Import the compiled CUDA extension

# Create a test tensor
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")

# Run the inclusive scan
y = cuda_scan.inclusive_scan(x)

print("Input:", x.cpu().numpy())  # [1, 2, 3, 4, 5]
print("Scan:", y.cpu().numpy())   # [1, 3, 6, 10, 15]
