import torch
import time
import math
import operator
from typing import Callable, Union, Tuple, Optional

def naive_scan_batched(arr: torch.Tensor, op: Callable = operator.add, 
                       identity_element: Union[int, float] = 0, dim: int = 1) -> torch.Tensor:
    """
    Naive sequential exclusive scan implementation for batched multi-dimensional tensors.
    
    Args:
        arr: Input tensor of shape (B, L, D) where B is batch size, L is sequence length, D is channels
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Exclusive scan result with same shape as input
    """
    result = torch.full_like(arr, identity_element)
    
    # Get the size of the dimension to scan over
    dim_size = arr.shape[dim]
    
    # Handle the special case for the first element
    if dim_size > 0:
        # Keep the first position as identity_element (already set in initialization)
        pass
    
    # Process the rest of the elements sequentially
    for i in range(1, dim_size):
        # Create slices for the current and previous positions
        slice_current = [slice(None)] * arr.ndim
        slice_current[dim] = i
        
        slice_prev = [slice(None)] * arr.ndim
        slice_prev[dim] = i-1
        
        # Apply the operation between the previous result and previous input
        result[tuple(slice_current)] = op(result[tuple(slice_prev)], arr[tuple(slice_prev)])
    
    return result

def blelloch_scan_batched(arr: torch.Tensor, op: Callable = operator.add, 
                         identity_element: Union[int, float] = 0, dim: int = 1) -> torch.Tensor:
    """
    Blelloch parallel exclusive scan implementation for batched multi-dimensional tensors.
    
    Args:
        arr: Input tensor of shape (B, L, D) where B is batch size, L is sequence length, D is channels
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Exclusive scan result with same shape as input
    """
    # Clone the input to avoid modifying it
    result = arr.clone()
    
    # Get the original shape
    original_shape = result.shape
    original_dim_size = original_shape[dim]
    
    # Return early for empty tensors or dimension size <= 1
    if arr.numel() == 0 or original_dim_size <= 1:
        # For empty tensor or dim_size=0, return as is
        # For dim_size=1, the exclusive scan is just the identity
        if original_dim_size == 1:
            return torch.full_like(arr, identity_element)
        return result
    
    # Move the dimension to scan to the end for easier processing
    if dim != -1 and dim != result.ndim - 1:
        result = result.movedim(dim, -1)
    
    # Reshape to 2D: (-1, L) where -1 combines all other dimensions
    flat_shape = (-1, result.shape[-1])
    result = result.reshape(flat_shape)
    
    # Get the number of rows and the sequence length
    n_rows, seq_len = result.shape
    
    # Round up sequence length to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the sequence dimension if needed
    if seq_len < pow2:
        padding = torch.full((n_rows, pow2 - seq_len), identity_element, 
                            device=result.device, dtype=result.dtype)
        result = torch.cat((result, padding), dim=1)
        padded_len = pow2
    else:
        padded_len = seq_len
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(padded_len))):
        step = 2 ** (d+1)
        
        # Create indices for the update
        idx_target = torch.arange(step-1, padded_len, step, device=result.device)
        idx_source = idx_target - step//2
        
        # Perform the operation
        result[:, idx_target] = op(result[:, idx_target], result[:, idx_source])
    
    # Set the last element to identity_element (for exclusive scan)
    result[:, -1] = identity_element
    
    # Down-sweep phase
    for d in range(int(math.log2(padded_len))-1, -1, -1):
        step = 2 ** (d+1)
        
        # Create indices for the update
        idx_right = torch.arange(step-1, padded_len, step, device=result.device)
        idx_left = idx_right - step//2
        
        # Save the left values before they're overwritten
        temp = result[:, idx_left].clone()
        
        # Update left with right
        result[:, idx_left] = result[:, idx_right]
        
        # Update right with combined value
        result[:, idx_right] = op(result[:, idx_right], temp)
    
    # Remove the padding
    result = result[:, :seq_len]
    
    # Reshape back to the original shape
    if dim != -1 and dim != arr.ndim - 1:
        result = result.reshape(original_shape[:-1] + (original_dim_size,))
        result = result.movedim(-1, dim)
    else:
        result = result.reshape(original_shape)
    
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

def verify_batched_scan(arr: torch.Tensor, op: Callable = operator.add, 
                       identity_element: Optional[Union[int, float]] = None, 
                       dim: int = 1, device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive implementation.
    
    Args:
        arr: Input tensor of shape (B, L, D)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator
        dim: Dimension along which to perform the scan
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Move tensor to specified device
    if not isinstance(arr, torch.Tensor):
        tensor_arr = torch.tensor(arr, device=device)
    else:
        tensor_arr = arr.to(device)
    
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
    
    # Check sample results for a random batch and channel
    if tensor_arr.ndim >= 3:
        b, d = 0, 0  # Sample first batch and channel for printing
        if dim == 1:
            print(f"Sample input (batch={b}, channel={d}): {tensor_arr[b, :, d].cpu().numpy()}")
            print(f"Sample naive result: {naive_result[b, :, d].cpu().numpy()}")
            print(f"Sample Blelloch result: {blelloch_result[b, :, d].cpu().numpy()}")
        else:
            print("Sample results cannot be shown for this dimension configuration.")
    
    # Check if results are equal (with a small tolerance for floating point differences)
    are_equal = torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5)
    
    if are_equal:
        print("✓ Verification PASSED: Both implementations produce identical results.")
    else:
        # Calculate max difference
        max_diff = torch.max(torch.abs(naive_result - blelloch_result))
        print(f"✗ Verification FAILED: Maximum difference between implementations: {max_diff.item()}")
    
    return are_equal

def benchmark_batched_scan(arr: torch.Tensor, op: Callable = operator.add, 
                          identity_element: Optional[Union[int, float]] = None, 
                          dim: int = 1, iterations: int = 20, 
                          device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the batched scan implementations.
    
    Args:
        arr: Input tensor of shape (B, L, D)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator
        dim: Dimension along which to perform the scan
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Move tensor to specified device
    if not isinstance(arr, torch.Tensor):
        tensor_arr = torch.tensor(arr, device=device)
    else:
        tensor_arr = arr.to(device)
    
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
    print(f"Input tensor shape: {tensor_arr.shape}")
    print(f"Scan dimension: {dim}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different batch sizes, sequence lengths, and channel dimensions
print("==== Testing and Verification with Batched Data ====")

# Create test tensors with shape (B, L, D)
batch_sizes = [2, 5]
seq_lengths = [8, 16, 32]
channel_dims = [3, 10]

for B in batch_sizes:
    for L in seq_lengths:
        for D in channel_dims:
            print(f"\n===== Testing with B={B}, L={L}, D={D} =====")
            # Create a test tensor with shape (B, L, D)
            test_tensor = torch.rand((B, L, D), device=device)
            
            # Verify scan along sequence dimension (dim=1)
            print("\n--- Scanning along sequence dimension (dim=1) ---")
            verify_batched_scan(test_tensor, operator.add, 0, dim=1, device=device)
            
            # Benchmark
            benchmark_batched_scan(test_tensor, operator.add, 0, dim=1, device=device)

# Test different scan dimensions
print("\n==== Testing Scan Along Different Dimensions ====")
test_tensor = torch.rand((4, 16, 8), device=device)

for dim in range(3):
    print(f"\n--- Scanning along dimension {dim} ---")
    verify_batched_scan(test_tensor, operator.add, 0, dim=dim, device=device)
    benchmark_batched_scan(test_tensor, operator.add, 0, dim=dim, device=device)

# Test with larger tensors for performance comparison
print("\n==== Performance Test with Larger Tensors ====")
large_tensor = torch.rand((32, 128, 64), device=device)
benchmark_batched_scan(large_tensor, operator.add, 0, dim=1, device=device)

# Test with different operators
print("\n==== Testing Different Operators on Batched Data ====")
test_tensor = torch.rand((4, 16, 8), device=device) * 10  # Scale for better max/min testing

operators = [
    (operator.add, "Addition", 0),
    (operator.mul, "Multiplication", 1),
    (max, "Maximum", float('-inf')),
    (min, "Minimum", float('inf'))
]

for op, name, identity in operators:
    print(f"\n--- Operator: {name} ---")
    verify_batched_scan(test_tensor, op, identity, dim=1, device=device)
    benchmark_batched_scan(test_tensor, op, identity, dim=1, device=device)