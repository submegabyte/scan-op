import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

T = TypeVar('T')

def default_getter(arr: torch.Tensor, idx: Tuple) -> torch.Tensor:
    """Default getter function for tensors."""
    return arr[idx]

def default_setter(arr: torch.Tensor, idx: Tuple, val: Any) -> None:
    """Default setter function for tensors."""
    arr[idx] = val

def tuple_getter(arr: Tuple[torch.Tensor, ...], idx: Tuple) -> Tuple[torch.Tensor, ...]:
    """Getter function for tuple of tensors."""
    return tuple(tensor[idx] for tensor in arr)

def tuple_setter(arr: Tuple[torch.Tensor, ...], idx: Tuple, val: Tuple[torch.Tensor, ...]) -> None:
    """Setter function for tuple of tensors."""
    for i, tensor in enumerate(arr):
        tensor[idx] = val[i]

def get_accessor_functions(arr: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Tuple[Callable, Callable]:
    """
    Returns appropriate getter and setter functions based on input type.
    
    Args:
        arr: Input tensor or tuple of tensors
        
    Returns:
        Tuple of (getter_function, setter_function)
    """
    if isinstance(arr, tuple) and all(isinstance(x, torch.Tensor) for x in arr):
        return tuple_getter, tuple_setter
    elif isinstance(arr, torch.Tensor):
        return default_getter, default_setter
    else:
        raise TypeError("Unsupported input type. Must be a tensor or tuple of tensors.")

def naive_scan_batched(arr: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
                      op: Callable = operator.add, 
                      identity_element: Union[int, float, Tuple] = 0, 
                      dim: int = 1,
                      getter: Optional[Callable] = None,
                      setter: Optional[Callable] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Naive sequential exclusive scan implementation supporting batched data and multiple channels.
    
    Args:
        arr: Input tensor with shape (B, L, D) or tuple of such tensors where:
             B = batch size
             L = sequence length
             D = number of channels/features
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        getter: Custom function to get values from arr
        setter: Custom function to set values in result
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Determine accessor functions if not provided
    if getter is None or setter is None:
        getter, setter = get_accessor_functions(arr)
    
    # Create result structure similar to input
    if isinstance(arr, tuple):
        # For tuples, create a tuple of tensors with the same shapes
        result = tuple(torch.full_like(tensor, identity_element[i] if isinstance(identity_element, tuple) 
                                       else identity_element) for i, tensor in enumerate(arr))
    else:
        # For tensors, create a tensor with the same shape
        result = torch.full_like(arr, identity_element)
    
    # Get shape information (using the first element if it's a tuple)
    if isinstance(arr, tuple):
        shape = arr[0].shape
    else:
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
        curr_tuple = tuple(curr_slice)
        prev_tuple = tuple(prev_slice)
        
        # Get values using the getter function
        prev_result = getter(result, prev_tuple)
        prev_arr = getter(arr, prev_tuple)
        
        # Apply the operation and set the value using the setter function
        setter(result, curr_tuple, op(prev_result, prev_arr))
        
    return result

def blelloch_scan_batched(arr: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
                         op: Callable = operator.add, 
                         identity_element: Union[int, float, Tuple] = 0, 
                         dim: int = 1,
                         getter: Optional[Callable] = None,
                         setter: Optional[Callable] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Blelloch parallel exclusive scan implementation supporting batched data and multiple channels.
    
    Args:
        arr: Input tensor with shape (B, L, D) or tuple of such tensors where:
             B = batch size
             L = sequence length
             D = number of channels/features
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        getter: Custom function to get values from arr
        setter: Custom function to set values in result
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Determine accessor functions if not provided
    if getter is None or setter is None:
        getter, setter = get_accessor_functions(arr)
    
    # Clone the input to avoid modifying it
    if isinstance(arr, tuple):
        arr_clone = tuple(tensor.clone() for tensor in arr)
    else:
        arr_clone = arr.clone()
    
    # Get shape information (using the first element if it's a tuple)
    if isinstance(arr_clone, tuple):
        orig_shape = arr_clone[0].shape
    else:
        orig_shape = arr_clone.shape
    
    seq_len = orig_shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr_clone
    
    # Reshape to make the scan dimension the last dimension for easier processing
    # This is more complex for tuples, so we'll convert to a special representation
    perm = list(range(len(orig_shape)))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    
    if isinstance(arr_clone, tuple):
        # Permute each tensor in the tuple
        arr_permuted = tuple(tensor.permute(*perm) for tensor in arr_clone)
    else:
        arr_permuted = arr_clone.permute(*perm)
    
    # Get new shape after permutation
    if isinstance(arr_permuted, tuple):
        shape = arr_permuted[0].shape
    else:
        shape = arr_permuted.shape
    
    seq_len = shape[-1]
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the sequence dimension if needed
    original_seq_len = seq_len
    if seq_len < pow2:
        if isinstance(arr_permuted, tuple):
            padded_tensors = []
            for i, tensor in enumerate(arr_permuted):
                padding_shape = list(shape[:-1]) + [pow2 - seq_len]
                padding_value = identity_element[i] if isinstance(identity_element, tuple) else identity_element
                padding = torch.full(padding_shape, padding_value, device=tensor.device, dtype=tensor.dtype)
                padded_tensors.append(torch.cat((tensor, padding), dim=-1))
            arr_padded = tuple(padded_tensors)
        else:
            padding_shape = list(shape[:-1]) + [pow2 - seq_len]
            padding = torch.full(padding_shape, identity_element, device=arr_permuted.device, dtype=arr_permuted.dtype)
            arr_padded = torch.cat((arr_permuted, padding), dim=-1)
    else:
        arr_padded = arr_permuted
    
    seq_len = pow2
    
    # Combine all batch dimensions for parallel processing
    flat_shape = (-1, seq_len)
    if isinstance(arr_padded, tuple):
        original_shape = arr_padded[0].shape
        arr_flat = tuple(tensor.reshape(flat_shape) for tensor in arr_padded)
    else:
        original_shape = arr_padded.shape
        arr_flat = arr_padded.reshape(flat_shape)
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        device = arr_flat[0].device if isinstance(arr_flat, tuple) else arr_flat.device
        indices = torch.arange(0, seq_len, step, device=device)
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values
            for i in range(len(left_indices)):
                left_idx = left_indices[i].item()
                right_idx = right_indices[i].item()
                
                # Get values using the getter function (with appropriate indices)
                if isinstance(arr_flat, tuple):
                    left_vals = tuple(tensor[:, left_idx] for tensor in arr_flat)
                    right_vals = tuple(tensor[:, right_idx] for tensor in arr_flat)
                    
                    # Apply the operation and set the result
                    result = op(right_vals, left_vals)
                    for j, tensor in enumerate(arr_flat):
                        tensor[:, right_idx] = result[j]
                else:
                    left_val = arr_flat[:, left_idx]
                    right_val = arr_flat[:, right_idx]
                    arr_flat[:, right_idx] = op(right_val, left_val)
    
    # Set the last element to identity element (for exclusive scan)
    if isinstance(arr_flat, tuple):
        for i, tensor in enumerate(arr_flat):
            tensor[:, -1] = identity_element[i] if isinstance(identity_element, tuple) else identity_element
    else:
        arr_flat[:, -1] = identity_element
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len))-1, -1, -1):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        device = arr_flat[0].device if isinstance(arr_flat, tuple) else arr_flat.device
        indices = torch.arange(0, seq_len, step, device=device)
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values using a temporary tensor to avoid in-place modification issues
            for i in range(len(left_indices)):
                left_idx = left_indices[i].item()
                right_idx = right_indices[i].item()
                
                if isinstance(arr_flat, tuple):
                    # Store left values in temp
                    temp = tuple(tensor[:, left_idx].clone() for tensor in arr_flat)
                    
                    # Update left values with right values
                    for j, tensor in enumerate(arr_flat):
                        tensor[:, left_idx] = arr_flat[j][:, right_idx]
                    
                    # Apply operation to right values
                    right_vals = tuple(tensor[:, right_idx] for tensor in arr_flat)
                    result = op(right_vals, temp)
                    for j, tensor in enumerate(arr_flat):
                        tensor[:, right_idx] = result[j]
                else:
                    temp = arr_flat[:, left_idx].clone()
                    arr_flat[:, left_idx] = arr_flat[:, right_idx]
                    arr_flat[:, right_idx] = op(arr_flat[:, right_idx], temp)
    
    # Reshape back to original shape (excluding padding)
    if isinstance(arr_flat, tuple):
        arr_reshaped = tuple(tensor.reshape(original_shape)[..., :original_seq_len] for tensor in arr_flat)
    else:
        arr_reshaped = arr_flat.reshape(original_shape)[..., :original_seq_len]
    
    # Permute back to original dimension order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    
    if isinstance(arr_reshaped, tuple):
        result = tuple(tensor.permute(*inv_perm) for tensor in arr_reshaped)
    else:
        result = arr_reshaped.permute(*inv_perm)
    
    return result

def get_identity_element(op: Callable, arr_type: Union[torch.Tensor, Tuple]) -> Union[int, float, Tuple]:
    """
    Returns the identity element for common operators based on the input type.
    
    Args:
        op: The operator function
        arr_type: The type of input (tensor or tuple of tensors)
        
    Returns:
        The identity element for the operator
    """
    if isinstance(arr_type, tuple):
        # For tuple inputs, create a tuple of identity elements
        if op is operator.add:
            return tuple(0 for _ in range(len(arr_type)))
        elif op is operator.mul:
            return tuple(1 for _ in range(len(arr_type)))
        elif op is operator.and_:
            return tuple(1 for _ in range(len(arr_type)))  # For bitwise AND, identity is all 1s
        elif op is operator.or_:
            return tuple(0 for _ in range(len(arr_type)))  # For bitwise OR, identity is all 0s
        else:
            raise ValueError("Unknown operator. Please provide an identity element.")
    else:
        # For tensor inputs, use the standard identity elements
        if op is operator.add:
            return 0
        elif op is operator.mul:
            return 1
        elif op is operator.and_:
            return 1  # For bitwise AND, identity is all 1s
        elif op is operator.or_:
            return 0  # For bitwise OR, identity is all 0s
        else:
            raise ValueError("Unknown operator. Please provide an identity element.")

def tuple_add(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Element-wise addition for tuples of tensors."""
    return tuple(a[i] + b[i] for i in range(len(a)))

def tuple_mul(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Element-wise multiplication for tuples of tensors."""
    return tuple(a[i] * b[i] for i in range(len(a)))

def verify_scan_batched(batch_size: int, seq_len: int, channels: int, op: Callable = operator.add, 
                       identity_element: Optional[Union[int, float, Tuple]] = None, 
                       device: str = "cpu", use_tuple: bool = False) -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        device: "cpu" or "cuda" for GPU (if available)
        use_tuple: Whether to test with a tuple of tensors
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    if use_tuple:
        # Create a tuple of two random tensors with shape (batch_size, seq_len, channels)
        tensor_arr1 = torch.rand(batch_size, seq_len, channels, device=device)
        tensor_arr2 = torch.rand(batch_size, seq_len, channels, device=device)
        
        # Ensure values are small integers for better numerical stability in tests
        tensor_arr1 = (tensor_arr1 * 10).int().float()
        tensor_arr2 = (tensor_arr2 * 10).int().float()
        
        tensor_arr = (tensor_arr1, tensor_arr2)
        
        # Define the appropriate tuple operator
        if op is operator.add:
            tuple_op = tuple_add
        elif op is operator.mul:
            tuple_op = tuple_mul
        else:
            raise ValueError("Only add and mul operations are supported for tuple inputs.")
    else:
        # Create a single random tensor with shape (batch_size, seq_len, channels)
        tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
        
        # Ensure values are small integers for better numerical stability in tests
        tensor_arr = (tensor_arr * 10).int().float()
        tuple_op = op
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op, tensor_arr)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Run both implementations
    naive_result = naive_scan_batched(tensor_arr, tuple_op, identity_element)
    blelloch_result = blelloch_scan_batched(tensor_arr, tuple_op, identity_element)
    
    print(f"Input type: {'Tuple of Tensors' if use_tuple else 'Tensor'}")
    print(f"Input tensor{'s' if use_tuple else ''} shape: {tensor_arr[0].shape if use_tuple else tensor_arr.shape}")
    print(f"Operator: {op.__name__}")
    print(f"Identity element: {identity_element}")
    
    # Check shapes
    if use_tuple:
        print(f"Naive result shapes: {[t.shape for t in naive_result]}")
        print(f"Blelloch result shapes: {[t.shape for t in blelloch_result]}")
        
        # Check if results are equal for each tensor in the tuple
        all_equal = True
        for i in range(len(naive_result)):
            if not torch.allclose(naive_result[i], blelloch_result[i], rtol=1e-5, atol=1e-5):
                all_equal = False
                break
    else:
        print(f"Naive result shape: {naive_result.shape}")
        print(f"Blelloch result shape: {blelloch_result.shape}")
        
        # Check if results are equal
        all_equal = torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5)
    
    if all_equal:
        print("✓ Verification PASSED: Both implementations produce identical results.")
        
        # Print sample results for first batch, first channel
        print("\nSample results for first batch, first channel:")
        if use_tuple:
            for i, t in enumerate(tensor_arr):
                print(f"Input {i}: {t[0, :, 0].cpu().numpy()}")
            for i, t in enumerate(naive_result):
                print(f"Naive scan {i}: {t[0, :, 0].cpu().numpy()}")
            for i, t in enumerate(blelloch_result):
                print(f"Blelloch scan {i}: {t[0, :, 0].cpu().numpy()}")
        else:
            print(f"Input: {tensor_arr[0, :, 0].cpu().numpy()}")
            print(f"Naive scan: {naive_result[0, :, 0].cpu().numpy()}")
            print(f"Blelloch scan: {blelloch_result[0, :, 0].cpu().numpy()}")
        
        return True
    else:
        print(f"✗ Verification FAILED")
        
        # Find where differences occur
        if use_tuple:
            for i in range(len(naive_result)):
                diff = (naive_result[i] - blelloch_result[i]).abs()
                max_diff = diff.max().item()
                if max_diff > 1e-5:
                    max_diff_indices = torch.where(diff == max_diff)
                    print(f"Tensor {i} - Max difference: {max_diff}")
                    print(f"Max difference at indices: {max_diff_indices}")
                    
                    # Print sample values where the difference is largest
                    b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
                    print(f"At batch={b}, pos={l}, channel={c}:")
                    print(f"  Naive result: {naive_result[i][b, l, c].item()}")
                    print(f"  Blelloch result: {blelloch_result[i][b, l, c].item()}")
        else:
            diff = (naive_result - blelloch_result).abs()
            max_diff = diff.max().item()
            max_diff_indices = torch.where(diff == max_diff)
            
            print(f"Max difference: {max_diff}")
            print(f"Max difference at indices: {max_diff_indices}")
            
            # Print sample values where the difference is largest
            b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
            print(f"At batch={b}, pos={l}, channel={c}:")
            print(f"  Naive result: {naive_result[b, l, c].item()}")
            print(f"  Blelloch result: {blelloch_result[b, l, c].item()}")
        
        return False

def benchmark_scan_batched(batch_size: int, seq_len: int, channels: int, op: Callable = operator.add, 
                          identity_element: Optional[Union[int, float, Tuple]] = None, 
                          iterations: int = 10, device: str = "cpu", use_tuple: bool = False) -> Tuple[float, float]:
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
        use_tuple: Whether to test with a tuple of tensors
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    if use_tuple:
        # Create a tuple of two random tensors
        tensor_arr1 = torch.rand(batch_size, seq_len, channels, device=device)
        tensor_arr2 = torch.rand(batch_size, seq_len, channels, device=device)
        tensor_arr = (tensor_arr1, tensor_arr2)
        
        # Define the appropriate tuple operator
        if op is operator.add:
            tuple_op = tuple_add
        elif op is operator.mul:
            tuple_op = tuple_mul
        else:
            raise ValueError("Only add and mul operations are supported for tuple inputs.")
    else:
        # Create a single random tensor
        tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
        tuple_op = op
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op, tensor_arr)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Warmup
    for _ in range(5):
        naive_scan_batched(tensor_arr, tuple_op, identity_element)
        blelloch_scan_batched(tensor_arr, tuple_op, identity_element)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_batched(tensor_arr, tuple_op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_batched(tensor_arr, tuple_op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results for shape ({batch_size}, {seq_len}, {channels}) on {device.upper()} with {'Tuple' if use_tuple else 'Tensor'} type:")
    print(f"Operator: {op.__name__}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different batch sizes, sequence lengths, and channel dimensions
print("\n==== Testing and Verification with Batched Data (Standard Tensor) ====")

test_configs = [
    (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
    (4, 16, 5),   # Medium config: 4 batches, 16 length, 5 channels
    (8, 32, 2)    # Larger config: 8 batches, 32 length, 2 channels
]

for batch, length, channels in test_configs:
    print(f"\n=== Testing batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched(batch, length, channels, operator.add, device=device, use_tuple=False)

# Test with tuple inputs
print("\n==== Testing and Verification with Batched Data (Tuple of Tensors) ====")

for batch, length, channels in test_configs:
    print(f"\n=== Testing batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched(batch, length, channels, operator.add, device=device, use_tuple=True)

# Benchmark with different configurations
print("\n==== Performance Benchmarks with Batched Data (Standard Tensor) ====")

benchmark_configs = [
    (1, 100, 1),    # Single batch, medium sequence, single channel
    (10, 100, 1),   # Multiple batches, medium sequence, single channel
    (10, 100, 10),  # Multiple batches, medium sequence, multiple channels
    (32, 128, 64),  # Typical ML batch/sequence/feature configuration
]

operators = [
    (operator.add, "Addition"),
    (operator.mul, "Multiplication"),
]

for batch, length, channels in benchmark_configs:
    print(f"\n--- Benchmarking batch={batch}, seq_len={length}, channels={channels} ---")
    
    for op, name in operators:
        identity = get_identity_element(op, torch.tensor(0))
        print(f"\nOperator: {name}")
        benchmark_scan_batched(batch, length, channels, op, identity, device=device, use_tuple=False)

# Benchmark with tuple inputs
print("\n==== Performance Benchmarks with Batched Data (Tuple of Tensors) ====")

for batch, length, channels in benchmark_configs[:2]:  # Use smaller configurations for tuple benchmarks
    print(f"\n--- Benchmarking batch={batch}, seq_len={length}, channels={channels} ---")
    
    for op, name in operators:
        # Create a dummy tuple to get the identity element
        dummy_tuple = (torch.tensor(0), torch.tensor(0))
        identity = get_identity_element(op, dummy_tuple)
        print(f"\nOperator: {name}")
        benchmark_scan_batched(batch, length, channels, op, identity, device=device, use_tuple=True)