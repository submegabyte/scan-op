import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, Protocol

# Define a protocol for indexable objects
class Indexable(Protocol):
    def __getitem__(self, idx: Any) -> Any:
        ...

# Custom operator for tensor tuples
def add_tensor_tuples(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    return tuple(a_i + b_i for a_i, b_i in zip(a, b))

def mul_tensor_tuples(a: Tuple[torch.Tensor, ...], b: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    return tuple(a_i * b_i for a_i, b_i in zip(a, b))

# Add the selective scan operation for 2-tuples
# From https://arxiv.org/pdf/2208.04933
def selective_scan_op(s: Tuple[torch.Tensor, torch.Tensor], 
                     c: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the selective scan operation from the paper.
    
    Args:
        s: A tuple of two tensors (sa, sb)
        c: A tuple of two tensors (ca, cb)
        
    Returns:
        A tuple of two tensors after applying the selective scan operation
    """
    sa, sb = s
    ca, cb = c

    sa_new = ca * sa
    sb_new = ca * sb + cb

    return (sa_new, sb_new)

# Custom indexing function to handle different types of indexable objects
def custom_index(obj: Any, idx: Any) -> Any:
    """
    Generic indexing function that handles different types of objects.
    
    Args:
        obj: The object to index (tensor, tuple of tensors, etc.)
        idx: The index to use
        
    Returns:
        The indexed object
    """
    if isinstance(obj, tuple) and all(isinstance(item, torch.Tensor) for item in obj):
        # For tuples of tensors, apply indexing to each tensor
        return tuple(item[idx] for item in obj)
    else:
        # For regular tensors or other indexable objects
        return obj[idx]

def custom_full_like(obj: Any, fill_value: Any) -> Any:
    """
    Creates a full-like object based on the input object type.
    
    Args:
        obj: The object to mimic (tensor, tuple of tensors, etc.)
        fill_value: The value to fill with
        
    Returns:
        An object of the same type as obj, filled with fill_value
    """
    if isinstance(obj, tuple) and all(isinstance(item, torch.Tensor) for item in obj):
        # For tuples of tensors, create a tuple of full_like tensors
        if isinstance(fill_value, tuple):
            return tuple(torch.full_like(item, val) for item, val in zip(obj, fill_value))
        else:
            return tuple(torch.full_like(item, fill_value) for item in obj)
    else:
        # For regular tensors
        return torch.full_like(obj, fill_value)

def naive_scan_batched(arr: Any, op: Callable = operator.add, 
                      identity_element: Any = 0, dim: int = 1) -> Any:
    """
    Naive sequential exclusive scan implementation supporting batched data, multiple channels,
    and custom indexable objects like tuples of tensors.
    
    Args:
        arr: Input object with shape (B, L, D) where:
             B = batch size
             L = sequence length
             D = number of channels/features
             Can be a tensor or tuple of tensors
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # For tuples of tensors, get shape from the first tensor
    if isinstance(arr, tuple) and all(isinstance(item, torch.Tensor) for item in arr):
        shape = arr[0].shape
    else:
        shape = arr.shape
    
    result = custom_full_like(arr, identity_element)
    seq_len = shape[dim]
    
    # Create appropriate indexing for the scan dimension
    for i in range(1, seq_len):
        # Create slices for the current and previous positions
        curr_slice = [slice(None)] * len(shape)
        prev_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        prev_slice[dim] = i-1
        
        # Apply the operation using custom indexing
        curr_idx = tuple(curr_slice)
        prev_idx = tuple(prev_slice)
        
        # Get values using custom indexing
        prev_result = custom_index(result, prev_idx)
        prev_arr = custom_index(arr, prev_idx)
        
        # Apply the operation and update the result
        if isinstance(result, tuple):
            # For tuples of tensors, update each tensor individually
            new_value = op(prev_result, prev_arr)
            for j in range(len(result)):
                result[j][curr_idx] = new_value[j]
        else:
            # For regular tensors
            result[curr_idx] = op(prev_result, prev_arr)
        
    return result

def blelloch_scan_batched(arr: Any, op: Callable = operator.add, 
                         identity_element: Any = 0, dim: int = 1) -> Any:
    """
    Blelloch parallel exclusive scan implementation supporting batched data, multiple channels,
    and custom indexable objects like tuples of tensors.
    
    Args:
        arr: Input object with shape (B, L, D) where:
             B = batch size
             L = sequence length
             D = number of channels/features
             Can be a tensor or tuple of tensors
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # For tuples of tensors, we need to handle each tensor separately
    if isinstance(arr, tuple) and all(isinstance(item, torch.Tensor) for item in arr):
        # Clone each tensor to avoid modifying the input
        arr_tensors = tuple(item.clone() for item in arr)
        device = arr_tensors[0].device
        dtype = arr_tensors[0].dtype
        shape = arr_tensors[0].shape
    else:
        # Clone the input to avoid modifying it
        arr_tensors = arr.clone()
        device = arr_tensors.device
        dtype = arr_tensors.dtype
        shape = arr_tensors.shape
    
    # Get shape information
    orig_shape = shape
    seq_len = orig_shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr_tensors
    
    # Reshape to make the scan dimension the last dimension for easier processing
    # This is more complex for tuples of tensors
    perm = list(range(len(orig_shape)))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    
    if isinstance(arr_tensors, tuple):
        arr_tensors = tuple(item.permute(*perm) for item in arr_tensors)
        # Get new shape after permutation
        shape = arr_tensors[0].shape
    else:
        arr_tensors = arr_tensors.permute(*perm)
        # Get new shape after permutation
        shape = arr_tensors.shape
    
    seq_len = shape[-1]
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the sequence dimension if needed
    original_seq_len = seq_len
    if seq_len < pow2:
        if isinstance(arr_tensors, tuple):
            padding_shape = list(shape[:-1]) + [pow2 - seq_len]
            
            # Create padding for each tensor in the tuple
            if isinstance(identity_element, tuple):
                paddings = tuple(torch.full(padding_shape, val, device=device, dtype=item.dtype) 
                               for item, val in zip(arr_tensors, identity_element))
            else:
                paddings = tuple(torch.full(padding_shape, identity_element, device=device, dtype=item.dtype) 
                               for item in arr_tensors)
            
            # Concatenate each tensor with its padding
            arr_tensors = tuple(torch.cat((item, padding), dim=-1) 
                              for item, padding in zip(arr_tensors, paddings))
        else:
            padding_shape = list(shape[:-1]) + [pow2 - seq_len]
            padding = torch.full(padding_shape, identity_element, device=device, dtype=dtype)
            arr_tensors = torch.cat((arr_tensors, padding), dim=-1)
        
        seq_len = pow2
    
    # Combine all batch dimensions for parallel processing
    flat_shape = (-1, seq_len)
    
    if isinstance(arr_tensors, tuple):
        original_shape = arr_tensors[0].shape
        arr_tensors = tuple(item.reshape(flat_shape) for item in arr_tensors)
    else:
        original_shape = arr_tensors.shape
        arr_tensors = arr_tensors.reshape(flat_shape)
    
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
            if isinstance(arr_tensors, tuple):
                # For tuples of tensors, apply the operation to each pair of tensors
                left_vals = tuple(item[:, left_indices] for item in arr_tensors)
                right_vals = tuple(item[:, right_indices] for item in arr_tensors)
                
                # Apply the operation
                result = op(left_vals, right_vals)
                
                # Update the right values
                for i, item in enumerate(arr_tensors):
                    item[:, right_indices] = result[i]
            else:
                # For regular tensors
                arr_tensors[:, right_indices] = op(arr_tensors[:, right_indices], arr_tensors[:, left_indices])
    
    # Set the last element to identity element (for exclusive scan)
    if isinstance(arr_tensors, tuple):
        if isinstance(identity_element, tuple):
            for item, val in zip(arr_tensors, identity_element):
                item[:, -1] = val
        else:
            for item in arr_tensors:
                item[:, -1] = identity_element
    else:
        arr_tensors[:, -1] = identity_element
    
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
            
            # Update values using temporary values to avoid in-place modification issues
            if isinstance(arr_tensors, tuple):
                # For tuples of tensors
                # Store temporary left values
                temp_left = tuple(item[:, left_indices].clone() for item in arr_tensors)
                
                # Set left values to right values
                for item in arr_tensors:
                    item[:, left_indices] = item[:, right_indices]
                
                # Get right values
                right_vals = tuple(item[:, right_indices] for item in arr_tensors)
                
                # Apply operation and update right values
                result = op(right_vals, temp_left)
                
                # Update the right values
                for i, item in enumerate(arr_tensors):
                    item[:, right_indices] = result[i]
            else:
                # For regular tensors
                temp = arr_tensors[:, left_indices].clone()
                arr_tensors[:, left_indices] = arr_tensors[:, right_indices]
                arr_tensors[:, right_indices] = op(arr_tensors[:, right_indices], temp)
    
    # Reshape back to original shape (excluding padding)
    if isinstance(arr_tensors, tuple):
        arr_tensors = tuple(item.reshape(original_shape)[..., :original_seq_len] for item in arr_tensors)
    else:
        arr_tensors = arr_tensors.reshape(original_shape)[..., :original_seq_len]
    
    # Permute back to original dimension order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    
    if isinstance(arr_tensors, tuple):
        arr_tensors = tuple(item.permute(*inv_perm) for item in arr_tensors)
    else:
        arr_tensors = arr_tensors.permute(*inv_perm)
    
    return arr_tensors

def get_identity_element(op: Callable, input_obj: Any = None) -> Any:
    """
    Returns the identity element for common operators, adapting to input type.
    
    Args:
        op: The operator function
        input_obj: Sample input object to determine the identity element structure
        
    Returns:
        The identity element for the operator, with appropriate structure
    """
    if isinstance(input_obj, tuple) and all(isinstance(item, torch.Tensor) for item in input_obj):
        # For tuples of tensors
        if op is add_tensor_tuples:
            return tuple(0 for _ in input_obj)
        elif op is mul_tensor_tuples:
            return tuple(1 for _ in input_obj)
        elif op is selective_scan_op:
            # For selective scan, the identity element is (1, 0)
            # This follows from the paper where sa_new = ca * sa and sb_new = ca * sb + cb
            # If ca=1 and cb=0, the operation preserves the original values
            return (1, 0)
        else:
            raise ValueError("Unknown operator for tensor tuples. Please provide an identity element.")
    else:
        # For regular tensors
        if op is operator.add:
            return 0
        elif op is operator.mul:
            return 1
        elif op is operator.and_:
            return 1  # For bitwise AND, identity is all 1s
        elif op is operator.or_:
            return 0  # For bitwise OR, identity is all 0s
        elif op is torch.min:
            return float('inf')
        else:
            raise ValueError("Unknown operator. Please provide an identity element.")

def verify_scan_batched(batch_size: int, seq_len: int, channels: int, op: Callable = operator.add, 
                       identity_element: Optional[Any] = None, 
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
        use_tuple: Whether to use a tuple of tensors instead of a single tensor
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    if use_tuple:
        # Create a tuple of random tensors
        num_tensors = 2  # Using 2 tensors in the tuple for simplicity
        tensor_arr = tuple(
            torch.rand(batch_size, seq_len, channels, device=device) 
            for _ in range(num_tensors)
        )
        
        # Ensure values are small integers for better numerical stability in tests
        tensor_arr = tuple((item * 10).int().float() for item in tensor_arr)
        
        # Use appropriate operators for tensor tuples
        if op is operator.add:
            op = add_tensor_tuples
        elif op is operator.mul:
            op = mul_tensor_tuples
        # No need to change op if it's already selective_scan_op
    else:
        # Create random tensor with shape (batch_size, seq_len, channels)
        tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
        
        # Ensure values are small integers for better numerical stability in tests
        tensor_arr = (tensor_arr * 10).int().float()
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op, tensor_arr)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Run both implementations
    naive_result = naive_scan_batched(tensor_arr, op, identity_element)
    blelloch_result = blelloch_scan_batched(tensor_arr, op, identity_element)
    
    print(f"Input tensor{'s' if use_tuple else ''} shape: {tuple(item.shape for item in tensor_arr) if use_tuple else tensor_arr.shape}")
    print(f"Operator: {op.__name__ if hasattr(op, '__name__') else op.__class__.__name__}")
    print(f"Identity element: {identity_element}")
    
    # Check shapes
    if use_tuple:
        print(f"Naive result shapes: {tuple(item.shape for item in naive_result)}")
        print(f"Blelloch result shapes: {tuple(item.shape for item in blelloch_result)}")
        
        # Check if results are equal for each tensor in the tuple
        results_equal = all(
            torch.allclose(naive_item, blelloch_item, rtol=1e-5, atol=1e-5)
            for naive_item, blelloch_item in zip(naive_result, blelloch_result)
        )
    else:
        print(f"Naive result shape: {naive_result.shape}")
        print(f"Blelloch result shape: {blelloch_result.shape}")
        
        # Check if results are equal
        results_equal = torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5)
    
    if results_equal:
        print("✓ Verification PASSED: Both implementations produce identical results.")
        
        # Print sample results for first batch, first channel
        print("\nSample results for first batch, first channel:")
        if use_tuple:
            for i, (input_tensor, naive_tensor, blelloch_tensor) in enumerate(zip(tensor_arr, naive_result, blelloch_result)):
                print(f"Tensor {i}:")
                print(f"Input: {input_tensor[0, :, 0].cpu().numpy()}")
                print(f"Naive scan: {naive_tensor[0, :, 0].cpu().numpy()}")
                print(f"Blelloch scan: {blelloch_tensor[0, :, 0].cpu().numpy()}")
        else:
            print(f"Input: {tensor_arr[0, :, 0].cpu().numpy()}")
            print(f"Naive scan: {naive_result[0, :, 0].cpu().numpy()}")
            print(f"Blelloch scan: {blelloch_result[0, :, 0].cpu().numpy()}")
        
        return True
    else:
        print(f"✗ Verification FAILED: Results are not identical")
        
        # Find differences for each tensor in the tuple
        if use_tuple:
            for i, (naive_tensor, blelloch_tensor) in enumerate(zip(naive_result, blelloch_result)):
                diff = (naive_tensor - blelloch_tensor).abs()
                max_diff = diff.max().item()
                max_diff_indices = torch.where(diff == max_diff)
                
                print(f"Tensor {i} - Max difference: {max_diff}")
                if max_diff > 0:
                    print(f"Max difference at indices: {max_diff_indices}")
                    
                    # Print sample values where the difference is largest
                    b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
                    print(f"At batch={b}, pos={l}, channel={c}:")
                    print(f"  Naive result: {naive_tensor[b, l, c].item()}")
                    print(f"  Blelloch result: {blelloch_tensor[b, l, c].item()}")
        else:
            # Find where differences occur
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
                          identity_element: Optional[Any] = None, 
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
        use_tuple: Whether to use a tuple of tensors instead of a single tensor
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    if use_tuple:
        # Create a tuple of random tensors
        num_tensors = 2  # Using 2 tensors in the tuple for simplicity
        tensor_arr = tuple(
            torch.rand(batch_size, seq_len, channels, device=device) 
            for _ in range(num_tensors)
        )
        
        # Use appropriate operators for tensor tuples
        if op is operator.add:
            op = add_tensor_tuples
        elif op is operator.mul:
            op = mul_tensor_tuples
        # No need to change op if it's already selective_scan_op
    else:
        # Create random tensor with shape (batch_size, seq_len, channels)
        tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op, tensor_arr)
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
    print(f"Operator: {op.__name__ if hasattr(op, '__name__') else op.__class__.__name__}")
    print(f"Using {'tuple of tensors' if use_tuple else 'single tensor'}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Add test and benchmark code for the selective scan operation
def test_selective_scan():
    print("\n==== Testing and Verification with Selective Scan Operation ====")
    
    # Test with different batch sizes, sequence lengths, and channel dimensions
    test_configs = [
        (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
        (4, 16, 5)    # Medium config: 4 batches, 16 length, 5 channels
    ]
    
    for batch, length, channels in test_configs:
        print(f"\n=== Testing selective scan with batch={batch}, seq_len={length}, channels={channels} ===")
        verify_scan_batched(batch, length, channels, selective_scan_op, device=device, use_tuple=True)
    
    print("\n==== Performance Benchmarks with Selective Scan Operation ====")
    
    benchmark_configs = [
        (1, 1000, 1),     # Single batch, long sequence, single channel
        (10, 100, 10),    # Multiple batches, medium sequence, multiple channels
    ]
    
    for batch, length, channels in benchmark_configs:
        print(f"\n--- Benchmarking selective scan with batch={batch}, seq_len={length}, channels={channels} ---")
        benchmark_scan_batched(batch, length, channels, selective_scan_op, None, device=device, use_tuple=True)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Run the selective scan tests
test_selective_scan()