import torch
import time
import math
import operator
import numpy as np
from typing import Callable, Union, List, Tuple, Optional

def naive_scan_batched(arr: Tuple[torch.Tensor, torch.Tensor], op: Callable, 
                      identity_element: Tuple[Union[int, float], Union[int, float]], dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive sequential exclusive scan implementation for batched inputs with 2-tensor-tuples.
    
    Args:
        arr: Input tuple of two tensors, each of shape [B, L, D] where:
             B = batch size, L = sequence length, D = number of channels
        op: Binary associative operator that takes two tuples of tensors and returns a tuple of tensors
        identity_element: Tuple of identity elements for the operator (one for each tensor)
        dim: Dimension along which to perform the scan (default: 1, sequence length)
        
    Returns:
        Exclusive scan result as a tuple of two tensors using the specified operator
    """
    # Unpack tuple input
    arr_0, arr_1 = arr
    identity_0, identity_1 = identity_element
    
    # Get sequence length along the specified dimension
    seq_len = arr_0.size(dim)
    
    # Initialize results with identity elements
    result_0 = torch.full_like(arr_0, identity_0)
    result_1 = torch.full_like(arr_1, identity_1)
    
    # For each position in the sequence (except the first which stays as identity)
    for i in range(1, seq_len):
        # Create indices to slice the appropriate dimension
        idx = [slice(None)] * arr_0.dim()
        idx[dim] = i - 1
        prev_idx = tuple(idx)
        
        idx[dim] = i
        curr_idx = tuple(idx)
        
        # Create tuples for the operation
        prev_result_tuple = (result_0[prev_idx], result_1[prev_idx])
        prev_arr_tuple = (arr_0[prev_idx], arr_1[prev_idx])
        
        # Apply the scan operation on tuples
        result_tuple = op(prev_result_tuple, prev_arr_tuple)
        
        # Unpack and store the results
        result_0[curr_idx], result_1[curr_idx] = result_tuple
    
    return result_0, result_1

def blelloch_scan_batched(arr: Tuple[torch.Tensor, torch.Tensor], op: Callable, 
                         identity_element: Tuple[Union[int, float], Union[int, float]], dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Blelloch parallel exclusive scan implementation for batched inputs with 2-tensor-tuples.
    
    Args:
        arr: Input tuple of two tensors, each of shape [B, L, D] where:
             B = batch size, L = sequence length, D = number of channels
        op: Binary associative operator that takes two tuples of tensors and returns a tuple of tensors
        identity_element: Tuple of identity elements for the operator (one for each tensor)
        dim: Dimension along which to perform the scan (default: 1, sequence length)
        
    Returns:
        Exclusive scan result as a tuple of two tensors using the specified operator
    """
    # Unpack tuple input
    arr_0, arr_1 = arr
    identity_0, identity_1 = identity_element
    
    # Make copies to avoid modifying the inputs
    arr_0 = arr_0.clone()
    arr_1 = arr_1.clone()
    
    # Handle empty input
    if arr_0.size(dim) == 0:
        return arr
    
    # Get sequence length along the specified dimension
    seq_len = arr_0.size(dim)
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the arrays if needed
    if seq_len < pow2:
        original_len = seq_len
        
        # Create padding shape
        pad_shape = list(arr_0.shape)
        pad_shape[dim] = pow2 - seq_len
        
        # Create padding tensors filled with identity elements
        padding_0 = torch.full(pad_shape, identity_0, device=arr_0.device, dtype=arr_0.dtype)
        padding_1 = torch.full(pad_shape, identity_1, device=arr_1.device, dtype=arr_1.dtype)
        
        # Create indices for concatenation
        dims = list(range(arr_0.dim()))
        dims.remove(dim)
        dims = [dim] + dims
        
        # Rearrange dimensions to put scan dimension first, then concatenate
        arr_0 = torch.cat([arr_0.permute(*dims), padding_0.permute(*dims)], dim=0)
        arr_1 = torch.cat([arr_1.permute(*dims), padding_1.permute(*dims)], dim=0)
        
        # Permute back to original dimension order
        inv_dims = list(range(arr_0.dim()))
        inv_dims.pop(0)
        inv_dims.insert(dim, 0)
        arr_0 = arr_0.permute(*inv_dims)
        arr_1 = arr_1.permute(*inv_dims)
        
        seq_len = pow2
    else:
        original_len = seq_len
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d + 1)
        
        for i in range(0, seq_len, step):
            # Create indices for the current positions
            left_idx = [slice(None)] * arr_0.dim()
            left_idx[dim] = i + step // 2 - 1
            left_idx = tuple(left_idx)
            
            right_idx = [slice(None)] * arr_0.dim()
            right_idx[dim] = i + step - 1
            right_idx = tuple(right_idx)
            
            # Apply the operation to the tuples
            right_tuple = (arr_0[right_idx], arr_1[right_idx])
            left_tuple = (arr_0[left_idx], arr_1[left_idx])
            result_tuple = op(right_tuple, left_tuple)
            
            # Unpack and store the results
            arr_0[right_idx], arr_1[right_idx] = result_tuple
    
    # Set the last elements to identity elements (for exclusive scan)
    last_idx = [slice(None)] * arr_0.dim()
    last_idx[dim] = seq_len - 1
    arr_0[tuple(last_idx)] = identity_0
    arr_1[tuple(last_idx)] = identity_1
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len)) - 1, -1, -1):
        step = 2 ** (d + 1)
        
        for i in range(0, seq_len, step):
            # Create indices for the current positions
            left_idx = [slice(None)] * arr_0.dim()
            left_idx[dim] = i + step // 2 - 1
            left_idx = tuple(left_idx)
            
            right_idx = [slice(None)] * arr_0.dim()
            right_idx[dim] = i + step - 1
            right_idx = tuple(right_idx)
            
            # Clone tensors at left index
            temp_0 = arr_0[left_idx].clone()
            temp_1 = arr_1[left_idx].clone()
            
            # Swap values
            arr_0[left_idx] = arr_0[right_idx]
            arr_1[left_idx] = arr_1[right_idx]
            
            # Combine using the operation
            right_tuple = (arr_0[right_idx], arr_1[right_idx])
            temp_tuple = (temp_0, temp_1)
            result_tuple = op(right_tuple, temp_tuple)
            
            # Unpack and store the results
            arr_0[right_idx], arr_1[right_idx] = result_tuple
    
    # Return only the originally sized result
    if original_len < pow2:
        slices = [slice(None)] * arr_0.dim()
        slices[dim] = slice(0, original_len)
        return arr_0[tuple(slices)], arr_1[tuple(slices)]
    
    return arr_0, arr_1

## https://arxiv.org/pdf/2208.04933
def selective_scan_op(s, c):
    """
    Selective scan operator from the paper 'Efficiently Modeling Long Sequences with 
    Structured State Spaces' (https://arxiv.org/pdf/2208.04933).
    
    Args:
        s: Tuple of state tensors (sa, sb)
        c: Tuple of input tensors (ca, cb)
        
    Returns:
        Updated state tuple
    """
    sa, sb = s
    ca, cb = c

    sa = ca * sa
    sb = ca * sb + cb

    return sa, sb

def selective_scan_identity():
    """Return identity elements for selective scan operator."""
    return (1, 0)  # Identity for the selective scan (multiplicative identity for first tensor, additive for second)

def get_identity_element_tuple(op: Callable) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Returns the identity element tuple for common operators.
    
    Args:
        op: The operator function name (as string) or a reference to a predefined tuple operator
        
    Returns:
        A tuple of identity elements for the operator
    """
    # Check if the operator is a string name
    if isinstance(op, str):
        if op == "tuple_add":
            return (0, 0)
        elif op == "tuple_mul":
            return (1, 1)
        elif op == "tuple_max":
            return (float('-inf'), float('-inf'))
        elif op == "tuple_min":
            return (float('inf'), float('inf'))
        elif op == "selective_scan_op":
            return (1, 0)
        else:
            raise ValueError(f"Unknown operator name: {op}. Please provide identity elements.")
    else:
        # Try to determine from the operator object (if it's one of our predefined functions)
        # This is a simplified approach and might not work for all custom operators
        op_name = op.__name__ if hasattr(op, "__name__") else str(op)
        if "add" in op_name:
            return (0, 0)
        elif "mul" in op_name:
            return (1, 1)
        elif "max" in op_name:
            return (float('-inf'), float('-inf'))
        elif "min" in op_name:
            return (float('inf'), float('inf'))
        elif "selective_scan" in op_name:
            return (1, 0)
        else:
            raise ValueError("Unknown operator. Please provide identity elements.")

def verify_scan_batched_tuple(tensor_arr: Tuple[torch.Tensor, torch.Tensor], op: Callable, 
                             identity_element: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                             dim: int = 1) -> bool:
    """
    Verifies the batched Blelloch scan against the naive scan for 2-tensor-tuples.
    
    Args:
        tensor_arr: Input tuple of two tensors, each of shape [B, L, D]
        op: Binary associative operator for tuples
        identity_element: Tuple of identity elements for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 1, sequence length)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element_tuple(op)
        except ValueError:
            raise ValueError("Please provide identity elements for the custom tuple operator.")
    
    # Extract tensors from the tuple
    tensor_0, tensor_1 = tensor_arr
    
    # Run both implementations
    naive_result_0, naive_result_1 = naive_scan_batched(tensor_arr, op, identity_element, dim)
    blelloch_result_0, blelloch_result_1 = blelloch_scan_batched(tensor_arr, op, identity_element, dim)
    
    print(f"Input tensor shapes: {tensor_0.shape}, {tensor_1.shape}")
    print(f"Operator: {op.__name__ if hasattr(op, '__name__') else str(op)}")
    print(f"Identity elements: {identity_element}")
    print(f"Scan dimension: {dim}")
    
    # Print a small sample for verification
    if tensor_0.dim() == 3:
        print("\nSample results for batch 0, channel 0:")
        print(f"Input tensor 0: {tensor_0[0, :, 0].cpu().numpy()}")
        print(f"Input tensor 1: {tensor_1[0, :, 0].cpu().numpy()}")
        print(f"Naive scan tensor 0: {naive_result_0[0, :, 0].cpu().numpy()}")
        print(f"Naive scan tensor 1: {naive_result_1[0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan tensor 0: {blelloch_result_0[0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan tensor 1: {blelloch_result_1[0, :, 0].cpu().numpy()}")
    
    # Check if results are equal for both tensors (with a small tolerance for floating point differences)
    if (torch.allclose(naive_result_0, blelloch_result_0, rtol=1e-5, atol=1e-5) and 
        torch.allclose(naive_result_1, blelloch_result_1, rtol=1e-5, atol=1e-5)):
        print("✓ Verification PASSED: Both implementations produce identical results for both tensors.")
        return True
    else:
        # Compute the mean absolute error for both tensors
        mae_0 = torch.abs(naive_result_0 - blelloch_result_0).mean().item()
        mae_1 = torch.abs(naive_result_1 - blelloch_result_1).mean().item()
        print(f"✗ Verification FAILED:")
        print(f"Mean absolute error for tensor 0: {mae_0}")
        print(f"Mean absolute error for tensor 1: {mae_1}")
        
        # Find indices of maximum difference for tensor 0
        max_diff_idx_0 = torch.abs(naive_result_0 - blelloch_result_0).argmax()
        flat_idx_0 = max_diff_idx_0.item()
        multi_idx_0 = np.unravel_index(flat_idx_0, naive_result_0.shape)
        print(f"Maximum difference for tensor 0 at index {multi_idx_0}:")
        print(f"Naive value: {naive_result_0[multi_idx_0].item()}")
        print(f"Blelloch value: {blelloch_result_0[multi_idx_0].item()}")
        
        # Find indices of maximum difference for tensor 1
        max_diff_idx_1 = torch.abs(naive_result_1 - blelloch_result_1).argmax()
        flat_idx_1 = max_diff_idx_1.item()
        multi_idx_1 = np.unravel_index(flat_idx_1, naive_result_1.shape)
        print(f"Maximum difference for tensor 1 at index {multi_idx_1}:")
        print(f"Naive value: {naive_result_1[multi_idx_1].item()}")
        print(f"Blelloch value: {blelloch_result_1[multi_idx_1].item()}")
        
        return False

def benchmark_scan_batched_tuple(tensor_arr: Tuple[torch.Tensor, torch.Tensor], op: Callable, 
                                identity_element: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                                iterations: int = 20, dim: int = 1, device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the batched scan implementations for 2-tensor-tuples.
    
    Args:
        tensor_arr: Input tuple of two tensors, each of shape [B, L, D]
        op: Binary associative operator for tuples
        identity_element: Tuple of identity elements for the operator (defaults to None and will be determined automatically)
        iterations: Number of iterations for timing
        dim: Dimension along which to perform the scan (default: 1, sequence length)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Move tensors to the specified device
    tensor_0, tensor_1 = tensor_arr
    tensor_0 = tensor_0.to(device)
    tensor_1 = tensor_1.to(device)
    tensor_arr = (tensor_0, tensor_1)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element_tuple(op)
        except ValueError:
            raise ValueError("Please provide identity elements for the custom tuple operator.")
    
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
    
    print(f"\nBenchmark results on {device.upper()} with tuple operator (averaged over {iterations} iterations):")
    print(f"Tensor shapes: {tensor_0.shape}, {tensor_1.shape}")
    print(f"Scan dimension: {dim}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Define tuple operators
def tuple_add(tuple1, tuple2):
    """Add corresponding elements of two tuples."""
    return (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1])

def tuple_mul(tuple1, tuple2):
    """Multiply corresponding elements of two tuples."""
    return (tuple1[0] * tuple2[0], tuple1[1] * tuple2[1])

def tuple_max(tuple1, tuple2):
    """Element-wise maximum of two tuples."""
    return (torch.max(tuple1[0], tuple2[0]), torch.max(tuple1[1], tuple2[1]))

def tuple_min(tuple1, tuple2):
    """Element-wise minimum of two tuples."""
    return (torch.min(tuple1[0], tuple2[0]), torch.min(tuple1[1], tuple2[1]))

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different shapes and operators
print("\n==== Testing Batched Scan Implementation for 2-Tensor-Tuples ====")

# Create test tensors with batch and channel dimensions
batch_size = 8
seq_length = 16
channels = 4

# Create test tensors [B, L, D]
test_tensor_0 = torch.randint(1, 10, (batch_size, seq_length, channels)).float().to(device)
test_tensor_1 = torch.randint(1, 10, (batch_size, seq_length, channels)).float().to(device)
test_tuple = (test_tensor_0, test_tensor_1)

# Test with addition (sum scan)
print("\n=== Addition (Sum) ===")
verify_scan_batched_tuple(test_tuple, tuple_add, (0, 0), dim=1)

# Test with multiplication (product scan)
print("\n=== Multiplication (Product) ===")
verify_scan_batched_tuple(test_tuple, tuple_mul, (1, 1), dim=1)

# Test with maximum
print("\n=== Maximum ===")
verify_scan_batched_tuple(test_tuple, tuple_max, (float('-inf'), float('-inf')), dim=1)

# Test with selective scan operator
print("\n=== Selective Scan Operator ===")
verify_scan_batched_tuple(test_tuple, selective_scan_op, (1, 0), dim=1)

# Test scanning along different dimensions
print("\n=== Scanning Along Batch Dimension (dim=0) ===")
verify_scan_batched_tuple(test_tuple, tuple_add, (0, 0), dim=0)

print("\n=== Scanning Along Channel Dimension (dim=2) ===")
verify_scan_batched_tuple(test_tuple, tuple_add, (0, 0), dim=2)

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
    
    tensor_0 = torch.rand((B, L, D), device=device)
    tensor_1 = torch.rand((B, L, D), device=device)
    tensor_tuple = (tensor_0, tensor_1)
    
    # Benchmark with addition
    print("\nOperator: Tuple Addition")
    benchmark_scan_batched_tuple(tensor_tuple, tuple_add, (0, 0), iterations=10, dim=1, device=device)
    
    # Benchmark with multiplication
    print("\nOperator: Tuple Multiplication")
    benchmark_scan_batched_tuple(tensor_tuple, tuple_mul, (1, 1), iterations=10, dim=1, device=device)
    
    # Benchmark with selective scan
    print("\nOperator: Selective Scan")
    benchmark_scan_batched_tuple(tensor_tuple, selective_scan_op, (1, 0), iterations=10, dim=1, device=device)