import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional

def naive_scan_batched(arr: torch.Tensor, op: Callable = operator.add, 
                      identity_element: Union[int, float] = 0, dim: int = 1) -> torch.Tensor:
    """
    Naive sequential exclusive scan implementation supporting batched data and multiple channels.
    
    Args:
        arr: Input tensor with shape (B, L, D) where:
             B = batch size
             L = sequence length
             D = number of channels/features
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    result = torch.full_like(arr, identity_element)
    
    # Get shape information
    shape = arr.shape
    seq_len = shape[dim]
    
    # Create appropriate indexing for the scan dimension
    for i in range(1, seq_len):
        # Create slices for the current and previous positions
        curr_slice = [slice(None)] * len(shape)
        prev_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        prev_slice[dim] = i-1
        
        # Apply the operation
        result[tuple(curr_slice)] = op(result[tuple(prev_slice)], arr[tuple(prev_slice)])
        
    return result

def blelloch_scan_batched(arr: torch.Tensor, op: Callable = operator.add, 
                         identity_element: Union[int, float] = 0, dim: int = 1) -> torch.Tensor:
    """
    Blelloch parallel exclusive scan implementation supporting batched data and multiple channels.
    
    Args:
        arr: Input tensor with shape (B, L, D) where:
             B = batch size
             L = sequence length
             D = number of channels/features
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Clone the input to avoid modifying it
    arr = arr.clone()
    
    # Get shape information
    orig_shape = arr.shape
    seq_len = orig_shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr
    
    # Reshape to make the scan dimension the last dimension for easier processing
    # This simplifies the indexing operations
    perm = list(range(len(orig_shape)))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    arr = arr.permute(*perm)
    
    # Get new shape after permutation
    shape = arr.shape
    seq_len = shape[-1]
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the sequence dimension if needed
    original_seq_len = seq_len
    if seq_len < pow2:
        padding_shape = list(shape[:-1]) + [pow2 - seq_len]
        padding = torch.full(padding_shape, identity_element, device=arr.device, dtype=arr.dtype)
        arr = torch.cat((arr, padding), dim=-1)
        seq_len = pow2
    
    # Combine all batch dimensions for parallel processing
    flat_shape = (-1, seq_len)
    original_shape = arr.shape
    arr = arr.reshape(flat_shape)
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = torch.arange(0, seq_len, step, device=arr.device)
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values
            arr[:, right_indices] = op(arr[:, right_indices], arr[:, left_indices])
    
    # Set the last element to identity element (for exclusive scan)
    arr[:, -1] = identity_element
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len))-1, -1, -1):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = torch.arange(0, seq_len, step, device=arr.device)
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values using a temporary tensor to avoid in-place modification issues
            temp = arr[:, left_indices].clone()
            arr[:, left_indices] = arr[:, right_indices]
            arr[:, right_indices] = op(arr[:, right_indices], temp)
    
    # Reshape back to original shape (excluding padding)
    arr = arr.reshape(original_shape)[..., :original_seq_len]
    
    # Permute back to original dimension order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    arr = arr.permute(*inv_perm)
    
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

def verify_scan_batched(batch_size: int, seq_len: int, channels: int, op: Callable = operator.add, 
                       identity_element: Optional[Union[int, float]] = None, 
                       device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create random tensor with shape (batch_size, seq_len, channels)
    tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
    
    # Ensure values are small integers for better numerical stability in tests
    tensor_arr = (tensor_arr * 10).int().float()
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Run both implementations
    naive_result = naive_scan_batched(tensor_arr, op, identity_element)
    blelloch_result = blelloch_scan_batched(tensor_arr, op, identity_element)
    
    print(f"Input tensor shape: {tensor_arr.shape}")
    print(f"Operator: {op.__name__}")
    print(f"Identity element: {identity_element}")
    
    # Check shapes
    print(f"Naive result shape: {naive_result.shape}")
    print(f"Blelloch result shape: {blelloch_result.shape}")
    
    # Check if results are equal
    if torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5):
        print("✓ Verification PASSED: Both implementations produce identical results.")
        
        # Print sample results for first batch, first channel
        print("\nSample results for first batch, first channel:")
        print(f"Input: {tensor_arr[0, :, 0].cpu().numpy()}")
        print(f"Naive scan: {naive_result[0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan: {blelloch_result[0, :, 0].cpu().numpy()}")
        
        return True
    else:
        # Find where differences occur
        diff = (naive_result - blelloch_result).abs()
        max_diff = diff.max().item()
        max_diff_indices = torch.where(diff == max_diff)
        
        print(f"✗ Verification FAILED: Max difference: {max_diff}")
        print(f"Max difference at indices: {max_diff_indices}")
        
        # Print sample values where the difference is largest
        b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
        print(f"At batch={b}, pos={l}, channel={c}:")
        print(f"  Naive result: {naive_result[b, l, c].item()}")
        print(f"  Blelloch result: {blelloch_result[b, l, c].item()}")
        
        return False

def benchmark_scan_batched(batch_size: int, seq_len: int, channels: int, op: Callable = operator.add, 
                          identity_element: Optional[Union[int, float]] = None, 
                          iterations: int = 10, device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the batched scan implementations.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Create random tensor with shape (batch_size, seq_len, channels)
    tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Warmup
    for _ in range(5):
        naive_scan_batched(tensor_arr, op, identity_element)
        blelloch_scan_batched(tensor_arr, op, identity_element)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_batched(tensor_arr, op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_batched(tensor_arr, op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results for shape ({batch_size}, {seq_len}, {channels}) on {device.upper()}:")
    print(f"Operator: {op.__name__}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different batch sizes, sequence lengths, and channel dimensions
print("\n==== Testing and Verification with Batched Data ====")

test_configs = [
    (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
    (4, 16, 5),   # Medium config: 4 batches, 16 length, 5 channels
    (8, 32, 2)    # Larger config: 8 batches, 32 length, 2 channels
]

for batch, length, channels in test_configs:
    print(f"\n=== Testing batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched(batch, length, channels, operator.add, device=device)

# Benchmark with different configurations
print("\n==== Performance Benchmarks with Batched Data ====")

benchmark_configs = [
    (1, 1000, 1),    # Single batch, long sequence, single channel
    (10, 100, 1),    # Multiple batches, medium sequence, single channel
    (10, 100, 10),   # Multiple batches, medium sequence, multiple channels
    (32, 128, 64),    # Typical ML batch/sequence/feature configuration
    (32, 512, 64),
    (32, 2048, 64),
]

operators = [
    (operator.add, "Addition"),
    (operator.mul, "Multiplication"),
    (torch.max, "Maximum"),
]

for batch, length, channels in benchmark_configs:
    print(f"\n--- Benchmarking batch={batch}, seq_len={length}, channels={channels} ---")
    
    for op, name in operators:
        identity = get_identity_element(op)
        print(f"\nOperator: {name}")
        benchmark_scan_batched(batch, length, channels, op, identity, device=device)