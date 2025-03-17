import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any

def tuple_add(a: Tuple[torch.Tensor, torch.Tensor], b: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Addition operator for 2-tensor tuples."""
    a1, a2 = a
    b1, b2 = b
    return (a1 + b1, a2 + b2)

def tuple_mul(a: Tuple[torch.Tensor, torch.Tensor], b: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multiplication operator for 2-tensor tuples."""
    a1, a2 = a
    b1, b2 = b
    return (a1 * b1, a2 * b2)

def tuple_max(a: Tuple[torch.Tensor, torch.Tensor], b: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Element-wise maximum operator for 2-tensor tuples."""
    a1, a2 = a
    b1, b2 = b
    return (torch.maximum(a1, b1), torch.maximum(a2, b2))

def tuple_min(a: Tuple[torch.Tensor, torch.Tensor], b: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Element-wise minimum operator for 2-tensor tuples."""
    a1, a2 = a
    b1, b2 = b
    return (torch.minimum(a1, b1), torch.minimum(a2, b2))

def selective_scan_op(s: Tuple[torch.Tensor, torch.Tensor], c: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selective scan operation as defined in https://arxiv.org/pdf/2208.04933
    
    Args:
        s: Tuple of tensors (sa, sb)
        c: Tuple of tensors (ca, cb)
        
    Returns:
        Tuple of updated tensors (sa, sb)
    """
    sa, sb = s
    ca, cb = c

    sa = ca * sa
    sb = ca * sb + cb

    return sa, sb

def get_tuple_identity_element(op: Callable, shape: Tuple, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the identity element for common tuple operators.
    
    Args:
        op: The operator function
        shape: Shape for creating identity tensors
        dtype: Data type for identity tensors
        device: Device for identity tensors
        
    Returns:
        A tuple of two tensors representing the identity element
    """
    if op is tuple_add:
        return (torch.zeros(shape, dtype=dtype, device=device), 
                torch.zeros(shape, dtype=dtype, device=device))
    elif op is tuple_mul:
        return (torch.ones(shape, dtype=dtype, device=device), 
                torch.ones(shape, dtype=dtype, device=device))
    elif op is selective_scan_op:
        return (torch.ones(shape, dtype=dtype, device=device), 
                torch.zeros(shape, dtype=dtype, device=device))
    elif op is tuple_max:
        return (torch.full(shape, float('-inf'), dtype=dtype, device=device),
                torch.full(shape, float('-inf'), dtype=dtype, device=device))
    elif op is tuple_min:
        return (torch.full(shape, float('inf'), dtype=dtype, device=device),
                torch.full(shape, float('inf'), dtype=dtype, device=device))
    else:
        raise ValueError("Unknown operator. Please provide an identity element.")

def naive_scan_batched(arr: Tuple[torch.Tensor, torch.Tensor], op: Callable = tuple_add, 
                      identity_element: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                      dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive sequential exclusive scan implementation supporting batched data and multiple channels.
    Works with tuples of 2 tensors.
    
    Args:
        arr: Tuple of two input tensors, each with shape (B, L, D) where:
             B = batch size
             L = sequence length
             D = number of channels/features
        op: Binary associative operator for tuples (default: tuple_add)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Tuple of two tensors containing the exclusive scan result using the specified operator
    """
    arr1, arr2 = arr
    
    # Determine shape and device information
    shape = arr1.shape
    dtype = arr1.dtype
    device = arr1.device
    
    # Get identity element if not provided
    if identity_element is None:
        # Get identity element with appropriate broadcasting shape
        elem_shape = list(shape)
        elem_shape[dim] = 1  # We'll broadcast along the scan dimension
        identity_element = get_tuple_identity_element(op, tuple(elem_shape), dtype, device)
    
    # Create result tensors filled with the identity element and broadcast along scan dimension
    ident1, ident2 = identity_element
    result1 = ident1.expand_as(arr1).clone()
    result2 = ident2.expand_as(arr2).clone()
    
    # Get shape information
    seq_len = shape[dim]
    
    # Create appropriate indexing for the scan dimension
    for i in range(1, seq_len):
        # Create slices for the current and previous positions
        curr_slice = [slice(None)] * len(shape)
        prev_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        prev_slice[dim] = i-1
        
        # Apply the operation
        prev_result = (result1[tuple(prev_slice)], result2[tuple(prev_slice)])
        prev_arr = (arr1[tuple(prev_slice)], arr2[tuple(prev_slice)])
        
        new_result = op(prev_result, prev_arr)
        result1[tuple(curr_slice)], result2[tuple(curr_slice)] = new_result
        
    return result1, result2

def blelloch_scan_batched(arr: Tuple[torch.Tensor, torch.Tensor], op: Callable = tuple_add, 
                         identity_element: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                         dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Blelloch parallel exclusive scan implementation supporting batched data and multiple channels.
    Works with tuples of 2 tensors.
    
    Args:
        arr: Tuple of two input tensors, each with shape (B, L, D) where:
             B = batch size
             L = sequence length
             D = number of channels/features
        op: Binary associative operator for tuples (default: tuple_add)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Tuple of two tensors containing the exclusive scan result using the specified operator
    """
    # Clone the inputs to avoid modifying them
    arr1, arr2 = arr
    arr1 = arr1.clone()
    arr2 = arr2.clone()
    
    # Get shape information
    orig_shape = arr1.shape
    dtype = arr1.dtype
    device = arr1.device
    seq_len = orig_shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr1, arr2
    
    # Get identity element if not provided
    if identity_element is None:
        elem_shape = list(orig_shape)
        elem_shape[dim] = 1  # We'll broadcast along the scan dimension
        identity_element = get_tuple_identity_element(op, tuple(elem_shape), dtype, device)
    
    # Reshape to make the scan dimension the last dimension for easier processing
    # This simplifies the indexing operations
    perm = list(range(len(orig_shape)))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    arr1 = arr1.permute(*perm)
    arr2 = arr2.permute(*perm)
    
    # Get new shape after permutation
    shape = arr1.shape
    seq_len = shape[-1]
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the sequence dimension if needed
    original_seq_len = seq_len
    if seq_len < pow2:
        padding_shape = list(shape[:-1]) + [pow2 - seq_len]
        
        # Get identity element with correct shape for padding
        ident1, ident2 = identity_element
        # Reshape identity elements to match padding shape
        padding1 = torch.full(padding_shape, 0, device=device, dtype=dtype)
        padding2 = torch.full(padding_shape, 0, device=device, dtype=dtype)
        
        # Add values from identity elements
        if ident1.item() != 0:
            padding1.fill_(ident1.item())
        if ident2.item() != 0:
            padding2.fill_(ident2.item())
            
        arr1 = torch.cat((arr1, padding1), dim=-1)
        arr2 = torch.cat((arr2, padding2), dim=-1)
        seq_len = pow2
    
    # Combine all batch dimensions for parallel processing
    flat_shape = (-1, seq_len)
    original_shape = arr1.shape
    arr1 = arr1.reshape(flat_shape)
    arr2 = arr2.reshape(flat_shape)
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = torch.arange(0, seq_len, step, device=device)
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values
            left_vals = (arr1[:, left_indices], arr2[:, left_indices])
            right_vals = (arr1[:, right_indices], arr2[:, right_indices])
            
            new_right = op(right_vals, left_vals)
            arr1[:, right_indices], arr2[:, right_indices] = new_right
    
    # Set the last element to identity element (for exclusive scan)
    ident1, ident2 = identity_element
    # We need a broadcasted version for the last element
    arr1[:, -1] = ident1.view(-1, *([1] * (arr1.dim() - 2)))
    arr2[:, -1] = ident2.view(-1, *([1] * (arr2.dim() - 2)))
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len))-1, -1, -1):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = torch.arange(0, seq_len, step, device=device)
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values using temporary tensors to avoid in-place modification issues
            temp1 = arr1[:, left_indices].clone()
            temp2 = arr2[:, left_indices].clone()
            
            arr1[:, left_indices] = arr1[:, right_indices]
            arr2[:, left_indices] = arr2[:, right_indices]
            
            left_vals = (arr1[:, right_indices], arr2[:, right_indices])
            right_vals = (temp1, temp2)
            
            new_right = op(left_vals, right_vals)
            arr1[:, right_indices], arr2[:, right_indices] = new_right
    
    # Reshape back to original shape (excluding padding)
    arr1 = arr1.reshape(original_shape)[..., :original_seq_len]
    arr2 = arr2.reshape(original_shape)[..., :original_seq_len]
    
    # Permute back to original dimension order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    arr1 = arr1.permute(*inv_perm)
    arr2 = arr2.permute(*inv_perm)
    
    return arr1, arr2

def verify_scan_batched(batch_size: int, seq_len: int, channels: int, op: Callable = tuple_add, 
                       identity_element: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                       device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan for tuple operations.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        op: Binary associative operator for tuples (default: tuple_add)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create random tensors with shape (batch_size, seq_len, channels)
    tensor_arr1 = torch.rand(batch_size, seq_len, channels, device=device)
    tensor_arr2 = torch.rand(batch_size, seq_len, channels, device=device)
    
    # Ensure values are small integers for better numerical stability in tests
    tensor_arr1 = (tensor_arr1 * 10).int().float()
    tensor_arr2 = (tensor_arr2 * 10).int().float()
    
    # Determine identity element if not provided
    if identity_element is None:
        elem_shape = (batch_size, 1, channels)  # Use broadcastable shape for scan dimension
        try:
            identity_element = get_tuple_identity_element(op, elem_shape, tensor_arr1.dtype, tensor_arr1.device)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Run both implementations
    naive_result1, naive_result2 = naive_scan_batched((tensor_arr1, tensor_arr2), op, identity_element)
    blelloch_result1, blelloch_result2 = blelloch_scan_batched((tensor_arr1, tensor_arr2), op, identity_element)
    
    print(f"Input tensor shape: {tensor_arr1.shape}")
    print(f"Operator: {op.__name__}")
    
    # Check shapes
    print(f"Naive result shape: {naive_result1.shape}, {naive_result2.shape}")
    print(f"Blelloch result shape: {blelloch_result1.shape}, {blelloch_result2.shape}")
    
    # Check if results are equal
    if (torch.allclose(naive_result1, blelloch_result1, rtol=1e-5, atol=1e-5) and 
        torch.allclose(naive_result2, blelloch_result2, rtol=1e-5, atol=1e-5)):
        print("✓ Verification PASSED: Both implementations produce identical results.")
        
        # Print sample results for first batch, first channel
        print("\nSample results for first batch, first channel:")
        print(f"Input1: {tensor_arr1[0, :, 0].cpu().numpy()}")
        print(f"Input2: {tensor_arr2[0, :, 0].cpu().numpy()}")
        print(f"Naive scan1: {naive_result1[0, :, 0].cpu().numpy()}")
        print(f"Naive scan2: {naive_result2[0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan1: {blelloch_result1[0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan2: {blelloch_result2[0, :, 0].cpu().numpy()}")
        
        return True
    else:
        # Find where differences occur
        diff1 = (naive_result1 - blelloch_result1).abs()
        diff2 = (naive_result2 - blelloch_result2).abs()
        max_diff1 = diff1.max().item()
        max_diff2 = diff2.max().item()
        max_diff = max(max_diff1, max_diff2)
        
        if max_diff1 >= max_diff2:
            max_diff_indices = torch.where(diff1 == max_diff1)
        else:
            max_diff_indices = torch.where(diff2 == max_diff2)
        
        print(f"✗ Verification FAILED: Max difference: {max_diff}")
        print(f"Max difference at indices: {max_diff_indices}")
        
        # Print sample values where the difference is largest
        b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
        print(f"At batch={b}, pos={l}, channel={c}:")
        print(f"  Naive result1: {naive_result1[b, l, c].item()}")
        print(f"  Naive result2: {naive_result2[b, l, c].item()}")
        print(f"  Blelloch result1: {blelloch_result1[b, l, c].item()}")
        print(f"  Blelloch result2: {blelloch_result2[b, l, c].item()}")
        
        return False

def benchmark_scan_batched(batch_size: int, seq_len: int, channels: int, op: Callable = tuple_add, 
                          identity_element: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                          iterations: int = 10, device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the batched scan implementations for tuple operations.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        op: Binary associative operator for tuples (default: tuple_add)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Create random tensors with shape (batch_size, seq_len, channels)
    tensor_arr1 = torch.rand(batch_size, seq_len, channels, device=device)
    tensor_arr2 = torch.rand(batch_size, seq_len, channels, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        elem_shape = (batch_size, 1, channels)  # Use broadcastable shape for scan dimension
        try:
            identity_element = get_tuple_identity_element(op, elem_shape, tensor_arr1.dtype, tensor_arr1.device)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Warmup
    for _ in range(5):
        naive_scan_batched((tensor_arr1, tensor_arr2), op, identity_element)
        blelloch_scan_batched((tensor_arr1, tensor_arr2), op, identity_element)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_batched((tensor_arr1, tensor_arr2), op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_batched((tensor_arr1, tensor_arr2), op, identity_element)
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
print("\n==== Testing and Verification with Batched Tuple Data ====")

test_configs = [
    (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
    (4, 16, 5),   # Medium config: 4 batches, 16 length, 5 channels
    (8, 32, 2)    # Larger config: 8 batches, 32 length, 2 channels
]

for batch, length, channels in test_configs:
    print(f"\n=== Testing batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched(batch, length, channels, tuple_add, device=device)

# Also test selective scan operation
print("\n=== Testing selective_scan_op ===")
for batch, length, channels in test_configs[:1]:  # Just test one configuration for selective scan
    print(f"\n=== Testing selective_scan_op with batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched(batch, length, channels, selective_scan_op, device=device)

# Benchmark with different configurations
print("\n==== Performance Benchmarks with Batched Tuple Data ====")

benchmark_configs = [
    (1, 1000, 1),    # Single batch, long sequence, single channel
    (10, 100, 1),    # Multiple batches, medium sequence, single channel
    (10, 100, 10),   # Multiple batches, medium sequence, multiple channels
    (32, 128, 64),   # Typical ML batch/sequence/feature configuration
]

operators = [
    (tuple_add, "Addition"),
    (tuple_mul, "Multiplication"),
    (tuple_max, "Maximum"),
    (selective_scan_op, "Selective Scan"),
]

for batch, length, channels in benchmark_configs[:2]:  # Just test a subset of configurations for brevity
    print(f"\n--- Benchmarking batch={batch}, seq_len={length}, channels={channels} ---")
    
    for op, name in operators:
        elem_shape = (batch, 1, channels)  # Use broadcastable shape for scan dimension
        identity = get_tuple_identity_element(op, elem_shape, torch.float32, torch.device(device))
        print(f"\nOperator: {name}")
        benchmark_scan_batched(batch, length, channels, op, identity, device=device)