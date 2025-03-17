import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

# Define a generic type for elements
T = TypeVar('T')

def naive_scan_batched(arr: Any, op: Callable = operator.add, 
                      identity_element: Any = 0, dim: int = 1,
                      get_item: Callable = lambda x, idx: x[idx],
                      set_item: Callable = lambda x, idx, val: x.__setitem__(idx, val)) -> Any:
    """
    Naive sequential exclusive scan implementation supporting batched data with custom indexing.
    
    Args:
        arr: Input data structure (tensor, tuple of tensors, etc.)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        get_item: Function to get an item at a specific index
        set_item: Function to set an item at a specific index
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # For tensors, create a result tensor filled with identity element
    if isinstance(arr, torch.Tensor):
        result = torch.full_like(arr, identity_element)
    else:
        # For other data structures, create a deep copy
        import copy
        result = copy.deepcopy(arr)
        # Initialize with identity elements
        for i in range(arr.shape[dim] if hasattr(arr, 'shape') else len(arr)):
            idx = [slice(None)] * (len(arr.shape) if hasattr(arr, 'shape') else dim+1)
            idx[dim] = i
            set_item(result, tuple(idx), identity_element)
    
    # Get shape information
    shape = arr.shape if hasattr(arr, 'shape') else (1,) * dim + (len(arr),)
    seq_len = shape[dim]
    
    # Create appropriate indexing for the scan dimension
    for i in range(1, seq_len):
        # Create slices for the current and previous positions
        curr_slice = [slice(None)] * len(shape)
        prev_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        prev_slice[dim] = i-1
        
        # Apply the operation
        curr_val = get_item(result, tuple(prev_slice))
        prev_val = get_item(arr, tuple(prev_slice))
        set_item(result, tuple(curr_slice), op(curr_val, prev_val))
        
    return result

def blelloch_scan_batched(arr: Any, op: Callable = operator.add, 
                         identity_element: Any = 0, dim: int = 1,
                         get_item: Callable = lambda x, idx: x[idx],
                         set_item: Callable = lambda x, idx, val: x.__setitem__(idx, val)) -> Any:
    """
    Blelloch parallel exclusive scan implementation supporting custom indexing.
    
    Args:
        arr: Input data structure (tensor, tuple of tensors, etc.)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        get_item: Function to get an item at a specific index
        set_item: Function to set an item at a specific index
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Clone/copy the input to avoid modifying it
    if isinstance(arr, torch.Tensor):
        arr_copy = arr.clone()
    else:
        import copy
        arr_copy = copy.deepcopy(arr)
    
    # Get shape information
    orig_shape = arr_copy.shape if hasattr(arr_copy, 'shape') else (1,) * dim + (len(arr_copy),)
    seq_len = orig_shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr_copy
    
    # For tensors, we can use permute to reshape
    if isinstance(arr_copy, torch.Tensor):
        # Reshape to make the scan dimension the last dimension for easier processing
        perm = list(range(len(orig_shape)))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        arr_copy = arr_copy.permute(*perm)
        shape = arr_copy.shape
        seq_len = shape[-1]
        
        # For non-tensor types, we'll need to handle the "permutation" differently
        # This would require custom implementation depending on the data structure
    else:
        # For simplicity, we'll assume the scan dimension is already the last dimension
        # for non-tensor types
        shape = orig_shape
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the sequence dimension if needed (for tensors)
    original_seq_len = seq_len
    if seq_len < pow2 and isinstance(arr_copy, torch.Tensor):
        padding_shape = list(shape[:-1]) + [pow2 - seq_len]
        padding = torch.full(padding_shape, identity_element, device=arr_copy.device, dtype=arr_copy.dtype)
        arr_copy = torch.cat((arr_copy, padding), dim=-1)
        seq_len = pow2
    elif seq_len < pow2:
        # For non-tensor types, we'd need a custom padding implementation
        # For this example, we'll just process the actual length without padding
        pass
    
    # For tensors, we can reshape for parallel processing
    if isinstance(arr_copy, torch.Tensor):
        flat_shape = (-1, seq_len)
        original_shape = arr_copy.shape
        arr_copy = arr_copy.reshape(flat_shape)
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len)) if seq_len & (seq_len - 1) == 0 else int(math.log2(pow2))):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        if isinstance(arr_copy, torch.Tensor):
            indices = torch.arange(0, seq_len, step, device=arr_copy.device)
        else:
            indices = range(0, seq_len, step)
            
        if len(indices) > 0:
            for idx in indices:
                left_idx = idx + step//2 - 1
                right_idx = idx + step - 1
                
                # Ensure indices are within bounds
                if right_idx < seq_len:
                    # Get values using custom indexing
                    if isinstance(arr_copy, torch.Tensor):
                        right_val = arr_copy[:, right_idx]
                        left_val = arr_copy[:, left_idx]
                        # Update values
                        arr_copy[:, right_idx] = op(right_val, left_val)
                    else:
                        # For non-tensor types, use the provided get_item and set_item functions
                        right_idx_tuple = tuple([slice(None)] * (dim) + [right_idx])
                        left_idx_tuple = tuple([slice(None)] * (dim) + [left_idx])
                        
                        right_val = get_item(arr_copy, right_idx_tuple)
                        left_val = get_item(arr_copy, left_idx_tuple)
                        
                        # Update values
                        set_item(arr_copy, right_idx_tuple, op(right_val, left_val))
    
    # Set the last element to identity element (for exclusive scan)
    if isinstance(arr_copy, torch.Tensor):
        arr_copy[:, -1] = identity_element
    else:
        last_idx = tuple([slice(None)] * (dim) + [seq_len - 1])
        set_item(arr_copy, last_idx, identity_element)
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len)) if seq_len & (seq_len - 1) == 0 else int(math.log2(pow2))-1, -1, -1):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        if isinstance(arr_copy, torch.Tensor):
            indices = torch.arange(0, seq_len, step, device=arr_copy.device)
        else:
            indices = range(0, seq_len, step)
            
        if len(indices) > 0:
            for idx in indices:
                left_idx = idx + step//2 - 1
                right_idx = idx + step - 1
                
                # Ensure indices are within bounds
                if right_idx < seq_len:
                    if isinstance(arr_copy, torch.Tensor):
                        # For tensors, use tensor operations
                        temp = arr_copy[:, left_idx].clone()
                        arr_copy[:, left_idx] = arr_copy[:, right_idx]
                        arr_copy[:, right_idx] = op(arr_copy[:, right_idx], temp)
                    else:
                        # For non-tensor types, use the provided get_item and set_item functions
                        right_idx_tuple = tuple([slice(None)] * (dim) + [right_idx])
                        left_idx_tuple = tuple([slice(None)] * (dim) + [left_idx])
                        
                        # Get values
                        right_val = get_item(arr_copy, right_idx_tuple)
                        left_val = get_item(arr_copy, left_idx_tuple)
                        
                        # Update values (need temporary storage to avoid overwriting)
                        temp = left_val
                        set_item(arr_copy, left_idx_tuple, right_val)
                        set_item(arr_copy, right_idx_tuple, op(right_val, temp))
    
    # For tensors, reshape back to original shape
    if isinstance(arr_copy, torch.Tensor):
        # Reshape back to original shape (excluding padding)
        arr_copy = arr_copy.reshape(original_shape)[..., :original_seq_len]
        
        # Permute back to original dimension order
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        arr_copy = arr_copy.permute(*inv_perm)
    
    return arr_copy

def get_identity_element(op: Callable, element_type: Any = None) -> Any:
    """
    Returns the identity element for common operators based on input type.
    
    Args:
        op: The operator function
        element_type: Optional hint about the type of elements being processed
        
    Returns:
        The identity element for the operator
    """
    # For tensor tuple types, return tuple of identity elements
    if element_type is not None and isinstance(element_type, tuple) and all(isinstance(x, torch.Tensor) for x in element_type):
        if op is operator.add:
            return tuple(torch.zeros_like(t) for t in element_type)
        elif op is operator.mul:
            return tuple(torch.ones_like(t) for t in element_type)
        elif op is tuple_max:
            return tuple(torch.full_like(t, float('-inf')) for t in element_type)
        elif op is tuple_min:
            return tuple(torch.full_like(t, float('inf')) for t in element_type)
    
    # For standard types
    if op is operator.add:
        return 0
    elif op is operator.mul:
        return 1
    elif op is operator.and_:
        return 1  # For bitwise AND, identity is all 1s
    elif op is operator.or_:
        return 0  # For bitwise OR, identity is all 0s
    elif op is torch.max or op is max:
        return float('-inf')
    elif op is torch.min or op is min:
        return float('inf')
    else:
        raise ValueError("Unknown operator. Please provide an identity element.")

# Custom operators for tensor tuples
def tuple_add(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Add corresponding tensors in two tuples element-wise."""
    return tuple(a_i + b_i for a_i, b_i in zip(a, b))

def tuple_mul(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Multiply corresponding tensors in two tuples element-wise."""
    return tuple(a_i * b_i for a_i, b_i in zip(a, b))

def tuple_max(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Take element-wise maximum of corresponding tensors in two tuples."""
    return tuple(torch.max(a_i, b_i) for a_i, b_i in zip(a, b))

def tuple_min(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Take element-wise minimum of corresponding tensors in two tuples."""
    return tuple(torch.min(a_i, b_i) for a_i, b_i in zip(a, b))

# Custom getters and setters for tensor tuples
def get_tuple_item(tuple_of_tensors: Tuple[torch.Tensor, ...], idx: Tuple) -> Tuple[torch.Tensor, ...]:
    """Get item from tuple of tensors at the specified index."""
    return tuple(t[idx] for t in tuple_of_tensors)

def set_tuple_item(tuple_of_tensors: List[torch.Tensor], idx: Tuple, value: Tuple[torch.Tensor, ...]) -> None:
    """Set item in tuple of tensors at the specified index."""
    for i, t in enumerate(tuple_of_tensors):
        t[idx] = value[i]

def create_tensor_tuple(batch_size: int, seq_len: int, channels_list: List[int], device: str = "cpu") -> Tuple[torch.Tensor, ...]:
    """
    Creates a tuple of random tensors for testing.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels_list: List of channel dimensions for each tensor in the tuple
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of random tensors
    """
    return tuple(
        torch.rand(batch_size, seq_len, channels, device=device) * 10
        for channels in channels_list
    )

def verify_scan_with_tensor_tuples(batch_size: int, seq_len: int, channels_list: List[int], 
                                  op: Callable = tuple_add, device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan for tensor tuples.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels_list: List of channel dimensions for each tensor in the tuple
        op: Binary associative operator for tuple operations
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create tuple of random tensors
    tensor_tuple = create_tensor_tuple(batch_size, seq_len, channels_list, device)
    
    # Convert to list for easier modification during scan
    tensor_list = list(tensor_tuple)
    
    # Get identity element
    identity_element = get_identity_element(op, tensor_tuple)
    
    # Run both implementations with custom getters and setters
    naive_result = naive_scan_batched(
        tensor_list, op, identity_element, dim=1,
        get_item=get_tuple_item, set_item=set_tuple_item
    )
    
    blelloch_result = blelloch_scan_batched(
        tensor_list, op, identity_element, dim=1,
        get_item=get_tuple_item, set_item=set_tuple_item
    )
    
    print(f"Input tensor tuple: {len(tensor_tuple)} tensors, each with shape {tensor_tuple[0].shape}")
    print(f"Operator: {op.__name__ if hasattr(op, '__name__') else op.__class__.__name__}")
    
    # Convert results to tuples for comparison
    naive_result_tuple = tuple(naive_result)
    blelloch_result_tuple = tuple(blelloch_result)
    
    # Check shapes
    print(f"Naive result: {len(naive_result_tuple)} tensors, each with shape {naive_result_tuple[0].shape}")
    print(f"Blelloch result: {len(blelloch_result_tuple)} tensors, each with shape {blelloch_result_tuple[0].shape}")
    
    # Check if results are equal
    all_equal = all(
        torch.allclose(naive_t, blelloch_t, rtol=1e-5, atol=1e-5)
        for naive_t, blelloch_t in zip(naive_result_tuple, blelloch_result_tuple)
    )
    
    if all_equal:
        print("✓ Verification PASSED: Both implementations produce identical results for tensor tuples.")
        
        # Print sample results for first batch, first channel of first tensor
        print("\nSample results for first batch, first channel of first tensor:")
        print(f"Input: {tensor_tuple[0][0, :, 0].cpu().numpy()}")
        print(f"Naive scan: {naive_result_tuple[0][0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan: {blelloch_result_tuple[0][0, :, 0].cpu().numpy()}")
        
        return True
    else:
        # Find where differences occur
        print("✗ Verification FAILED")
        for i, (naive_t, blelloch_t) in enumerate(zip(naive_result_tuple, blelloch_result_tuple)):
            diff = (naive_t - blelloch_t).abs()
            max_diff = diff.max().item()
            
            if max_diff > 1e-5:
                max_diff_indices = torch.where(diff == max_diff)
                
                print(f"Tensor {i} has max difference: {max_diff}")
                print(f"Max difference at indices: {max_diff_indices}")
                
                # Print sample values where the difference is largest
                b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
                print(f"At batch={b}, pos={l}, channel={c}:")
                print(f"  Naive result: {naive_result_tuple[i][b, l, c].item()}")
                print(f"  Blelloch result: {blelloch_result_tuple[i][b, l, c].item()}")
        
        return False

def benchmark_scan_with_tensor_tuples(batch_size: int, seq_len: int, channels_list: List[int], 
                                     op: Callable = tuple_add, 
                                     iterations: int = 5, device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the batched scan implementations for tensor tuples.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels_list: List of channel dimensions for each tensor in the tuple
        op: Binary associative operator for tuple operations
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Create tuple of random tensors
    tensor_tuple = create_tensor_tuple(batch_size, seq_len, channels_list, device)
    
    # Convert to list for easier modification during scan
    tensor_list = list(tensor_tuple)
    
    # Get identity element
    identity_element = get_identity_element(op, tensor_tuple)
    
    # Warmup
    for _ in range(2):
        naive_scan_batched(tensor_list, op, identity_element, dim=1, get_item=get_tuple_item, set_item=set_tuple_item)
        blelloch_scan_batched(tensor_list, op, identity_element, dim=1, get_item=get_tuple_item, set_item=set_tuple_item)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_batched(tensor_list, op, identity_element, dim=1, get_item=get_tuple_item, set_item=set_tuple_item)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_batched(tensor_list, op, identity_element, dim=1, get_item=get_tuple_item, set_item=set_tuple_item)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    op_name = op.__name__ if hasattr(op, '__name__') else op.__class__.__name__
    print(f"\nBenchmark results for tensor tuples with {len(channels_list)} tensors, each ({batch_size}, {seq_len}, varies) on {device.upper()}:")
    print(f"Operator: {op_name}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Example test code
if __name__ == "__main__":
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test with tensor tuples
    print("\n==== Testing and Verification with Tensor Tuples ====")
    
    test_configs = [
        (2, 8, [3, 5]),        # Small config: 2 batches, 8 length, 2 tensors with channels 3 and 5
        (4, 16, [5, 7, 3]),    # Medium config: 4 batches, 16 length, 3 tensors with varying channels
        (8, 32, [2, 4, 8, 16]) # Larger config: 8 batches, 32 length, 4 tensors with varying channels
    ]
    
    for batch, length, channels_list in test_configs:
        print(f"\n=== Testing batch={batch}, seq_len={length}, tensors={len(channels_list)} ===")
        verify_scan_with_tensor_tuples(batch, length, channels_list, tuple_add, device=device)
    
    # Benchmark with different configurations
    print("\n==== Performance Benchmarks with Tensor Tuples ====")
    
    benchmark_configs = [
        (1, 100, [10, 10]),           # Single batch, medium sequence, 2 tensors
        (10, 100, [8, 16, 32]),       # Multiple batches, medium sequence, 3 tensors
        (32, 128, [16, 32, 64, 128]), # Typical ML batch/sequence with 4 tensors
    ]
    
    operators = [
        (tuple_add, "TupleAddition"),
        (tuple_mul, "TupleMultiplication"),
        (tuple_max, "TupleMaximum"),
    ]
    
    for batch, length, channels_list in benchmark_configs:
        print(f"\n--- Benchmarking batch={batch}, seq_len={length}, tensors={len(channels_list)} ---")
        
        for op, name in operators:
            print(f"\nOperator: {name}")
            benchmark_scan_with_tensor_tuples(batch, length, channels_list, op, device=device)