import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional

def naive_scan_batched(arr: torch.Tensor, op: Callable = operator.add, 
                      identity_element: Union[int, float] = 0, dim: int = 1) -> torch.Tensor:
    """
    Naive sequential exclusive scan implementation for batched inputs.
    
    Args:
        arr: Input tensor of shape [B, L, D] where:
             B = batch size, L = sequence length, D = number of channels
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, sequence length)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    result = arr.clone()
    batch_size = arr.size(0) if arr.dim() > 1 else 1
    seq_len = arr.size(dim)
    
    # Initialize result with identity element
    result = torch.full_like(arr, identity_element)
    
    # For each position in the sequence (except the first which stays as identity)
    for i in range(1, seq_len):
        # Create indices to slice the appropriate dimension
        idx = [slice(None)] * arr.dim()
        idx[dim] = i - 1
        prev_idx = tuple(idx)
        
        idx[dim] = i
        curr_idx = tuple(idx)
        
        # Apply the scan operation
        result[curr_idx] = op(result[prev_idx], arr[prev_idx])
    
    return result

def blelloch_scan_batched(arr: torch.Tensor, op: Callable = operator.add, 
                         identity_element: Union[int, float] = 0, dim: int = 1) -> torch.Tensor:
    """
    Blelloch parallel exclusive scan implementation for batched inputs.
    
    Args:
        arr: Input tensor of shape [B, L, D] where:
             B = batch size, L = sequence length, D = number of channels
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, sequence length)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Make a copy to avoid modifying the input
    arr = arr.clone()
    
    # Handle empty input
    if arr.size(dim) == 0:
        return arr
    
    # Get sequence length along the specified dimension
    seq_len = arr.size(dim)
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the array if needed
    if seq_len < pow2:
        original_len = seq_len
        
        # Create padding shape
        pad_shape = list(arr.shape)
        pad_shape[dim] = pow2 - seq_len
        
        # Create padding tensor filled with identity element
        padding = torch.full(pad_shape, identity_element, device=arr.device, dtype=arr.dtype)
        
        # Create indices for concatenation
        dims = list(range(arr.dim()))
        dims.remove(dim)
        dims = [dim] + dims
        
        # Rearrange dimensions to put scan dimension first, then concatenate
        arr = torch.cat([arr.permute(*dims), padding.permute(*dims)], dim=0)
        
        # Permute back to original dimension order
        inv_dims = list(range(arr.dim()))
        inv_dims.pop(0)
        inv_dims.insert(dim, 0)
        arr = arr.permute(*inv_dims)
        
        seq_len = pow2
    else:
        original_len = seq_len
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d + 1)
        
        for i in range(0, seq_len, step):
            # Create indices for the current positions
            left_idx = [slice(None)] * arr.dim()
            left_idx[dim] = i + step // 2 - 1
            left_idx = tuple(left_idx)
            
            right_idx = [slice(None)] * arr.dim()
            right_idx[dim] = i + step - 1
            right_idx = tuple(right_idx)
            
            # Apply the operation
            arr[right_idx] = op(arr[right_idx], arr[left_idx])
    
    # Set the last element to identity element (for exclusive scan)
    last_idx = [slice(None)] * arr.dim()
    last_idx[dim] = seq_len - 1
    arr[tuple(last_idx)] = identity_element
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len)) - 1, -1, -1):
        step = 2 ** (d + 1)
        
        for i in range(0, seq_len, step):
            # Create indices for the current positions
            left_idx = [slice(None)] * arr.dim()
            left_idx[dim] = i + step // 2 - 1
            left_idx = tuple(left_idx)
            
            right_idx = [slice(None)] * arr.dim()
            right_idx[dim] = i + step - 1
            right_idx = tuple(right_idx)
            
            # Swap and then combine
            temp = arr[left_idx].clone()
            arr[left_idx] = arr[right_idx]
            arr[right_idx] = op(arr[right_idx], temp)
    
    # Return only the originally sized result
    if original_len < pow2:
        slices = [slice(None)] * arr.dim()
        slices[dim] = slice(0, original_len)
        return arr[tuple(slices)]
    return arr

def get_identity_element(op: Callable) -> Union[int, float]:
    """
    Returns the identity element for common operators.
    
    Args:
        op: The operator function
        
    Returns:
        The identity element for the operator
    """
    if op is operator.add:
        return 0
    elif op is operator.mul:
        return 1
    elif op is operator.and_:
        return 1  # For bitwise AND, identity is all 1s
    elif op is operator.or_:
        return 0  # For bitwise OR, identity is all 0s
    elif op is torch.max:
        return float('-inf')
    elif op is torch.min:
        return float('inf')
    else:
        raise ValueError("Unknown operator. Please provide an identity element.")

def verify_scan_batched(tensor_arr: torch.Tensor, op: Callable = operator.add, 
                       identity_element: Optional[Union[int, float]] = None, 
                       dim: int = 1) -> bool:
    """
    Verifies the batched Blelloch scan against the naive scan.
    
    Args:
        tensor_arr: Input tensor of shape [B, L, D]
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 1, sequence length)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Run both implementations
    naive_result = naive_scan_batched(tensor_arr, op, identity_element, dim)
    blelloch_result = blelloch_scan_batched(tensor_arr, op, identity_element, dim)
    
    print(f"Input tensor shape: {tensor_arr.shape}")
    print(f"Operator: {op.__name__}")
    print(f"Identity element: {identity_element}")
    print(f"Scan dimension: {dim}")
    
    # Print a small sample for verification
    if tensor_arr.dim() == 3:
        print("\nSample results for batch 0, channel 0:")
        print(f"Input: {tensor_arr[0, :, 0].cpu().numpy()}")
        print(f"Naive scan: {naive_result[0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan: {blelloch_result[0, :, 0].cpu().numpy()}")
    
    # Check if results are equal (with a small tolerance for floating point differences)
    if torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5):
        print("✓ Verification PASSED: Both implementations produce identical results.")
        return True
    else:
        # Compute the mean absolute error
        mae = torch.abs(naive_result - blelloch_result).mean().item()
        print(f"✗ Verification FAILED: Mean absolute error between implementations: {mae}")
        
        # Find indices of maximum difference
        max_diff_idx = torch.abs(naive_result - blelloch_result).argmax()
        flat_idx = max_diff_idx.item()
        multi_idx = np.unravel_index(flat_idx, naive_result.shape)
        print(f"Maximum difference at index {multi_idx}:")
        print(f"Naive value: {naive_result[multi_idx].item()}")
        print(f"Blelloch value: {blelloch_result[multi_idx].item()}")
        
        return False

def benchmark_scan_batched(tensor_arr: torch.Tensor, op: Callable = operator.add, 
                          identity_element: Optional[Union[int, float]] = None, 
                          iterations: int = 20, dim: int = 1, device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the batched scan implementations.
    
    Args:
        tensor_arr: Input tensor of shape [B, L, D]
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        iterations: Number of iterations for timing
        dim: Dimension along which to perform the scan (default: 1, sequence length)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Move tensor to the specified device
    tensor_arr = tensor_arr.to(device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Warmup
    for _ in range(5):
        naive_scan_batched(tensor_arr, op, identity_element, dim)
        blelloch_scan_batched(tensor_arr, op, identity_element, dim)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_batched(tensor_arr, op, identity_element, dim)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_batched(tensor_arr, op, identity_element, dim)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results on {device.upper()} with operator {op.__name__} (averaged over {iterations} iterations):")
    print(f"Tensor shape: {tensor_arr.shape}")
    print(f"Scan dimension: {dim}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Import for multi-dimensional indexing
import numpy as np

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different shapes and operators
print("\n==== Testing Batched Scan Implementation ====")

# Create test tensors with batch and channel dimensions
batch_size = 8
seq_length = 16
channels = 4

# Create a test tensor [B, L, D]
test_tensor = torch.randint(1, 10, (batch_size, seq_length, channels)).float().to(device)

# Test with addition (sum scan)
print("\n=== Addition (Sum) ===")
verify_scan_batched(test_tensor, operator.add, 0, dim=1)

# Test with multiplication (product scan)
print("\n=== Multiplication (Product) ===")
verify_scan_batched(test_tensor, operator.mul, 1, dim=1)

# Test with maximum
print("\n=== Maximum ===")
verify_scan_batched(test_tensor, torch.max, float('-inf'), dim=1)

# Test scanning along different dimensions
print("\n=== Scanning Along Batch Dimension (dim=0) ===")
verify_scan_batched(test_tensor, operator.add, 0, dim=0)

print("\n=== Scanning Along Channel Dimension (dim=2) ===")
verify_scan_batched(test_tensor, operator.add, 0, dim=2)

# Benchmark with different batch sizes and sequence lengths
print("\n==== Performance Benchmarks with Different Shapes ====")

# Test configurations
configs = [
    {"batch": 4, "seq_len": 128, "channels": 4},
    {"batch": 16, "seq_len": 128, "channels": 4},
    {"batch": 16, "seq_len": 512, "channels": 4},
]

for config in configs:
    B, L, D = config["batch"], config["seq_len"], config["channels"]
    print(f"\n--- Shape: [B={B}, L={L}, D={D}] ---")
    
    tensor = torch.rand((B, L, D), device=device)
    
    # Benchmark with addition
    print("\nOperator: Addition")
    benchmark_scan_batched(tensor, operator.add, 0, iterations=10, dim=1, device=device)
    
    # Benchmark with multiplication
    print("\nOperator: Multiplication")
    benchmark_scan_batched(tensor, operator.mul, 1, iterations=10, dim=1, device=device)