import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional

def naive_scan_multidim(tensor: torch.Tensor, 
                       dim: int = 0,
                       op: Callable = operator.add, 
                       identity_element: Union[int, float] = 0) -> torch.Tensor:
    """
    Naive sequential exclusive scan implementation for multi-dimensional tensors.
    
    Args:
        tensor: Input tensor of shape [..., L, D] or [B, ..., L, D]
        dim: Dimension along which to perform the scan (default: 0)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        
    Returns:
        Exclusive scan result along the specified dimension
    """
    result = torch.full_like(tensor, identity_element)
    
    # Get the size of the dimension we're scanning over
    dim_size = tensor.size(dim)
    
    # Create slices for each position along the scan dimension
    for i in range(1, dim_size):
        # Create slice indices for the current and previous positions
        prev_idx = [slice(None)] * tensor.dim()
        prev_idx[dim] = slice(0, i)
        
        curr_idx = [slice(None)] * tensor.dim()
        curr_idx[dim] = i
        
        # Update the current position based on the operator and previous values
        result[curr_idx] = op(result[curr_idx[0]], tensor[prev_idx[0]])
        
    return result

def blelloch_scan_multidim(tensor: torch.Tensor, 
                          dim: int = 0,
                          op: Callable = operator.add, 
                          identity_element: Union[int, float] = 0) -> torch.Tensor:
    """
    Blelloch parallel exclusive scan implementation for multi-dimensional tensors.
    
    Args:
        tensor: Input tensor of shape [..., L, D] or [B, ..., L, D]
        dim: Dimension along which to perform the scan (default: 0)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        
    Returns:
        Exclusive scan result along the specified dimension
    """
    # Clone the input tensor to avoid modifying it
    result = tensor.clone()
    
    # Handle empty tensor
    if tensor.numel() == 0:
        return result
    
    # Get the size of the dimension we're scanning over
    dim_size = tensor.size(dim)
    
    # Handle single-element case
    if dim_size <= 1:
        if dim_size == 1:
            # For single element, set to identity
            idx = [slice(None)] * tensor.dim()
            idx[dim] = 0
            result[idx] = identity_element
        return result
    
    # Calculate the next power of 2 greater than or equal to dim_size
    pow2 = 1
    while pow2 < dim_size:
        pow2 *= 2
    
    # Pad the tensor if necessary
    original_dim_size = dim_size
    if dim_size < pow2:
        padding_size = [0] * (2 * tensor.dim())
        padding_size[2 * (tensor.dim() - 1 - dim)] = pow2 - dim_size
        result = torch.nn.functional.pad(result, tuple(padding_size), value=identity_element)
        dim_size = pow2
    
    # For easier manipulation, move the scan dimension to the front
    result = result.transpose(0, dim)
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(dim_size))):
        step = 2 ** (d + 1)
        for i in range(0, dim_size, step):
            if i + step - 1 < dim_size and i + step//2 - 1 < dim_size:
                result[i + step - 1] = op(result[i + step - 1], result[i + step//2 - 1])
    
    # Set the last element to identity element for exclusive scan
    if dim_size > 0:
        result[dim_size - 1] = identity_element
    
    # Down-sweep phase
    for d in range(int(math.log2(dim_size)) - 1, -1, -1):
        step = 2 ** (d + 1)
        for i in range(0, dim_size, step):
            if i + step - 1 < dim_size and i + step//2 - 1 < dim_size:
                temp = result[i + step//2 - 1].clone()
                result[i + step//2 - 1] = result[i + step - 1]
                result[i + step - 1] = op(result[i + step - 1], temp)
    
    # Move the scan dimension back to its original position
    result = result.transpose(0, dim)
    
    # Trim back to original size if we padded
    if original_dim_size < pow2:
        trim_idx = [slice(None)] * tensor.dim()
        trim_idx[dim] = slice(0, original_dim_size)
        result = result[trim_idx]
    
    return result

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
    elif op is max:
        return float('-inf')
    elif op is min:
        return float('inf')
    else:
        raise ValueError("Unknown operator. Please provide an identity element.")

def verify_scan_multidim(tensor_shape: List[int], 
                        scan_dim: int = 0,
                        op: Callable = operator.add, 
                        identity_element: Optional[Union[int, float]] = None, 
                        device: str = "cpu") -> bool:
    """
    Verifies the multi-dimensional Blelloch scan against the naive scan.
    
    Args:
        tensor_shape: Shape of the tensor to create
        scan_dim: Dimension along which to perform the scan
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create a random tensor of the specified shape
    tensor = torch.rand(tensor_shape, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Run both scan implementations
    naive_result = naive_scan_multidim(tensor, scan_dim, op, identity_element)
    blelloch_result = blelloch_scan_multidim(tensor, scan_dim, op, identity_element)
    
    print(f"Input tensor shape: {tensor.shape}")
    print(f"Scan dimension: {scan_dim}")
    print(f"Operator: {op.__name__}")
    print(f"Identity element: {identity_element}")
    
    # Check if results are equal (with a small tolerance for floating point differences)
    if torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5):
        print("✓ Verification PASSED: Both implementations produce identical results.")
        return True
    else:
        max_diff = torch.max(torch.abs(naive_result - blelloch_result)).item()
        print(f"✗ Verification FAILED: Maximum difference between implementations: {max_diff}")
        return False

def benchmark_scan_multidim(tensor_shape: List[int], 
                           scan_dim: int = 0,
                           op: Callable = operator.add, 
                           identity_element: Optional[Union[int, float]] = None, 
                           iterations: int = 100, 
                           device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the multi-dimensional scan implementations.
    
    Args:
        tensor_shape: Shape of the tensor to create
        scan_dim: Dimension along which to perform the scan
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Create a random tensor of the specified shape
    tensor = torch.rand(tensor_shape, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Warmup
    for _ in range(10):
        naive_scan_multidim(tensor, scan_dim, op, identity_element)
        blelloch_scan_multidim(tensor, scan_dim, op, identity_element)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_multidim(tensor, scan_dim, op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_multidim(tensor, scan_dim, op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results on {device.upper()} (averaged over {iterations} iterations):")
    print(f"Tensor shape: {tensor_shape}, Scan dim: {scan_dim}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different tensor shapes
print("==== Testing Multi-dimensional Scans ====")

# Simple cases
verify_scan_multidim([10, 5], scan_dim=0, device=device)  # L=10, D=5
verify_scan_multidim([10, 5], scan_dim=1, device=device)  # L=5, D=10 (scan over channels)

# With batch dimension
verify_scan_multidim([3, 10, 5], scan_dim=1, device=device)  # B=3, L=10, D=5
verify_scan_multidim([3, 10, 5], scan_dim=0, device=device)  # Scan over batch dim
verify_scan_multidim([3, 10, 5], scan_dim=2, device=device)  # Scan over channel dim

# More complex cases
verify_scan_multidim([2, 3, 16, 8], scan_dim=2, device=device)  # Multi-dim with larger sequence

# Benchmark different tensor shapes and scan dimensions
print("\n==== Performance Benchmarks ====")

benchmark_shapes = [
    ([128, 64], 0),          # Sequence scan L=128, D=64
    ([128, 64], 1),          # Channel scan L=128, D=64
    ([16, 128, 64], 1),      # Batch=16, sequence scan
    ([16, 128, 64], 0),      # Scan over batch dimension
    ([8, 8, 128, 64], 2),    # Complex tensor, scan over sequence dim
]

for shape, dim in benchmark_shapes:
    print(f"\n--- Shape: {shape}, Scan dim: {dim} ---")
    benchmark_scan_multidim(shape, dim, iterations=20, device=device)