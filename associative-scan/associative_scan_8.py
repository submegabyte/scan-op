import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional

def naive_scan_multi_channel(arr: torch.Tensor, op: Callable = operator.add, 
                            identity_element: Union[int, float] = 0, dim: int = 0) -> torch.Tensor:
    """
    Naive sequential exclusive scan implementation for multi-channel data.
    
    Args:
        arr: Input tensor of shape [L, D] where L is sequence length and D is number of channels
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 0, the sequence dimension)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Get the shape of the input tensor
    shape = arr.shape
    seq_len = shape[dim]
    
    # Create result tensor filled with identity element
    result = torch.full_like(arr, identity_element)
    
    # Create indexing slices for each dimension
    # This allows us to select the appropriate slices regardless of tensor dimensionality
    for i in range(1, seq_len):
        # Select all previous elements along the scan dimension
        prev_slice = [slice(None)] * len(shape)
        prev_slice[dim] = i - 1
        
        # Select current position along the scan dimension
        curr_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        
        # Perform operation between previous result and previous input
        result[curr_slice] = op(result[prev_slice], arr[prev_slice])
        
    return result

def blelloch_scan_multi_channel(arr: torch.Tensor, op: Callable = operator.add, 
                               identity_element: Union[int, float] = 0, dim: int = 0) -> torch.Tensor:
    """
    Blelloch parallel exclusive scan implementation for multi-channel data.
    
    Args:
        arr: Input tensor of shape [L, D] where L is sequence length and D is number of channels
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 0, the sequence dimension)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Clone the input to avoid modifying it
    arr = arr.clone()
    
    # Get the shape of the input tensor
    shape = arr.shape
    seq_len = shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # If we need padding, create a new padded tensor
    if seq_len < pow2:
        original_seq_len = seq_len
        
        # Create padding shape (only pad along the scan dimension)
        pad_shape = list(shape)
        pad_shape[dim] = pow2 - seq_len
        
        # Create padding tensor filled with identity element
        padding = torch.full(pad_shape, identity_element, device=arr.device, dtype=arr.dtype)
        
        # Create a list of tensors to concatenate
        tensors_to_concat = [arr, padding]
        
        # Concatenate along the scan dimension
        arr = torch.cat(tensors_to_concat, dim=dim)
        seq_len = pow2
    else:
        original_seq_len = seq_len
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d+1)
        
        for i in range(0, seq_len, step):
            # Create index slices for the current and previous positions
            curr_slice = [slice(None)] * len(shape)
            curr_slice[dim] = i + step - 1
            
            prev_slice = [slice(None)] * len(shape)
            prev_slice[dim] = i + step//2 - 1
            
            # Update the current position with the operation result
            arr[curr_slice] = op(arr[curr_slice], arr[prev_slice])
    
    # Set the last element to identity element (for exclusive scan)
    last_slice = [slice(None)] * len(shape)
    last_slice[dim] = seq_len - 1
    arr[last_slice] = identity_element
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len))-1, -1, -1):
        step = 2 ** (d+1)
        
        for i in range(0, seq_len, step):
            # Create index slices for the positions we need
            mid_slice = [slice(None)] * len(shape)
            mid_slice[dim] = i + step//2 - 1
            
            end_slice = [slice(None)] * len(shape)
            end_slice[dim] = i + step - 1
            
            # Store temporary value
            temp = arr[mid_slice].clone()
            
            # Swap
            arr[mid_slice] = arr[end_slice]
            
            # Update
            arr[end_slice] = op(arr[end_slice], temp)
    
    # Return only the originally sized result
    if original_seq_len < seq_len:
        orig_slice = [slice(None)] * len(shape)
        orig_slice[dim] = slice(0, original_seq_len)
        return arr[orig_slice]
    else:
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
    elif op is max:
        return float('-inf')
    elif op is min:
        return float('inf')
    else:
        raise ValueError("Unknown operator. Please provide an identity element.")

def verify_scan_multi_channel(arr: torch.Tensor, op: Callable = operator.add, 
                             identity_element: Optional[Union[int, float]] = None, 
                             dim: int = 0, device: str = "cpu") -> bool:
    """
    Verifies the multi-channel Blelloch scan against the naive scan.
    
    Args:
        arr: Input tensor or array that will be converted to a tensor
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 0, the sequence dimension)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Convert to tensor if not already
    if not isinstance(arr, torch.Tensor):
        tensor_arr = torch.tensor(arr, dtype=torch.float32, device=device)
    else:
        tensor_arr = arr.to(device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    naive_result = naive_scan_multi_channel(tensor_arr, op, identity_element, dim)
    blelloch_result = blelloch_scan_multi_channel(tensor_arr, op, identity_element, dim)
    
    print(f"Input tensor shape: {tensor_arr.shape}")
    print(f"Operator: {op.__name__}")
    print(f"Identity element: {identity_element}")
    
    # Check if results are equal (with a small tolerance for floating point differences)
    equal = torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5)
    
    if equal:
        print("✓ Verification PASSED: Both implementations produce identical results.")
        return True
    else:
        # Compute difference statistics for debugging
        diff = naive_result - blelloch_result
        print(f"✗ Verification FAILED: Max difference: {torch.max(torch.abs(diff))}")
        print(f"  Mean difference: {torch.mean(torch.abs(diff))}")
        return False

def benchmark_scan_multi_channel(arr: torch.Tensor, op: Callable = operator.add, 
                                identity_element: Optional[Union[int, float]] = None, 
                                dim: int = 0, iterations: int = 100, 
                                device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the multi-channel scan implementations.
    
    Args:
        arr: Input tensor or array that will be converted to a tensor
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 0, the sequence dimension)
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Convert to tensor if not already
    if not isinstance(arr, torch.Tensor):
        tensor_arr = torch.tensor(arr, dtype=torch.float32, device=device)
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
        naive_scan_multi_channel(tensor_arr, op, identity_element, dim)
        blelloch_scan_multi_channel(tensor_arr, op, identity_element, dim)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_multi_channel(tensor_arr, op, identity_element, dim)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_multi_channel(tensor_arr, op, identity_element, dim)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results on {device.upper()} with operator {op.__name__} (averaged over {iterations} iterations):")
    print(f"Tensor shape: {tensor_arr.shape}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with multi-channel data
print("\n==== Testing Multi-Channel Scan ====")

# Create test data with shape [L, D]
L, D = 8, 3  # Sequence length, Number of channels
test_data = torch.rand((L, D), device=device)
print(f"Test data shape: {test_data.shape}")

# Print sample data
print("Sample data (first few entries):")
print(test_data[:4])

# Verify with different operators
print("\n=== Addition (Sum) ===")
verify_scan_multi_channel(test_data, operator.add, 0, dim=0, device=device)

print("\n=== Multiplication (Product) ===")
verify_scan_multi_channel(test_data, operator.mul, 1, dim=0, device=device)

print("\n=== Maximum ===")
verify_scan_multi_channel(test_data, max, float('-inf'), dim=0, device=device)

# Benchmark with different sized inputs
print("\n==== Performance Benchmarks with Multi-Channel Data ====")
seq_lengths = [32, 128, 512]
channel_sizes = [1, 16, 64]

for seq_len in seq_lengths:
    for channels in channel_sizes:
        print(f"\n--- Sequence Length: {seq_len}, Channels: {channels} ---")
        test_tensor = torch.rand((seq_len, channels), device=device)
        benchmark_scan_multi_channel(test_tensor, operator.add, 0, dim=0, iterations=10, device=device)

# Test scanning along the channel dimension (dim=1)
print("\n==== Testing Scan Along Channel Dimension ====")
test_data = torch.rand((16, 32), device=device)  # [seq_len, channels]
print(f"Test data shape: {test_data.shape}")
print("\n=== Addition (Sum) along channel dimension ===")
verify_scan_multi_channel(test_data, operator.add, 0, dim=1, device=device)
benchmark_scan_multi_channel(test_data, operator.add, 0, dim=1, iterations=10, device=device)