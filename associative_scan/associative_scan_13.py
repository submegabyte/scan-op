import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar, Protocol

# Define a protocol for indexable types
class Indexable(Protocol):
    def __getitem__(self, idx: Any) -> Any:
        ...
    
    def __setitem__(self, idx: Any, value: Any) -> None:
        ...

T = TypeVar('T', bound=Indexable)

# Define custom indexing operators
def get_element(obj: Any, idx: Any) -> Any:
    """Get an element from an object using the given index."""
    if isinstance(obj, tuple) and all(torch.is_tensor(t) for t in obj):
        # For tuple of tensors, return tuple of indexed tensors
        return tuple(t[idx] for t in obj)
    else:
        # For regular tensors and other indexable objects
        return obj[idx]

def set_element(obj: Any, idx: Any, value: Any) -> None:
    """Set an element in an object using the given index."""
    if isinstance(obj, tuple) and all(torch.is_tensor(t) for t in obj):
        # For tuple of tensors, set each tensor's value
        for i, t in enumerate(obj):
            t[idx] = value[i]
    else:
        # For regular tensors and other indexable objects
        obj[idx] = value

def create_full_like(obj: Any, fill_value: Any, device: str = None) -> Any:
    """Create a new object filled with the given value, with the same structure as obj."""
    if isinstance(obj, tuple) and all(torch.is_tensor(t) for t in obj):
        # For tuple of tensors, create tuple of filled tensors
        return tuple(torch.full_like(t, fill_value) for t in obj)
    elif torch.is_tensor(obj):
        # For regular tensors
        return torch.full_like(obj, fill_value)
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

def clone_obj(obj: Any) -> Any:
    """Create a clone of the object."""
    if isinstance(obj, tuple) and all(torch.is_tensor(t) for t in obj):
        # For tuple of tensors, clone each tensor
        return tuple(t.clone() for t in obj)
    elif torch.is_tensor(obj):
        # For regular tensors
        return obj.clone()
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

def shape_of(obj: Any) -> Tuple[int, ...]:
    """Get the shape of the object."""
    if isinstance(obj, tuple) and all(torch.is_tensor(t) for t in obj):
        # For tuple of tensors, return shape of first tensor
        # Assuming all tensors in the tuple have the same shape
        return obj[0].shape
    elif torch.is_tensor(obj):
        # For regular tensors
        return obj.shape
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

def reshape_obj(obj: Any, shape: Tuple[int, ...]) -> Any:
    """Reshape the object to the given shape."""
    if isinstance(obj, tuple) and all(torch.is_tensor(t) for t in obj):
        # For tuple of tensors, reshape each tensor
        return tuple(t.reshape(shape) for t in obj)
    elif torch.is_tensor(obj):
        # For regular tensors
        return obj.reshape(shape)
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

def permute_obj(obj: Any, dims: List[int]) -> Any:
    """Permute the dimensions of the object."""
    if isinstance(obj, tuple) and all(torch.is_tensor(t) for t in obj):
        # For tuple of tensors, permute each tensor
        return tuple(t.permute(*dims) for t in obj)
    elif torch.is_tensor(obj):
        # For regular tensors
        return obj.permute(*dims)
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

def cat_obj(obj1: Any, obj2: Any, dim: int = -1) -> Any:
    """Concatenate objects along the given dimension."""
    if (isinstance(obj1, tuple) and isinstance(obj2, tuple) and 
        all(torch.is_tensor(t) for t in obj1) and all(torch.is_tensor(t) for t in obj2)):
        # For tuple of tensors, concatenate corresponding tensors
        if len(obj1) != len(obj2):
            raise ValueError("Tuples must have the same length for concatenation")
        return tuple(torch.cat((t1, t2), dim=dim) for t1, t2 in zip(obj1, obj2))
    elif torch.is_tensor(obj1) and torch.is_tensor(obj2):
        # For regular tensors
        return torch.cat((obj1, obj2), dim=dim)
    else:
        raise TypeError(f"Unsupported object types: {type(obj1)}, {type(obj2)}")

def naive_scan_batched(arr: Any, op: Callable = operator.add, 
                      identity_element: Any = 0, dim: int = 1) -> Any:
    """
    Naive sequential exclusive scan implementation supporting batched data, multiple channels,
    and custom data structures with custom indexing.
    
    Args:
        arr: Input data with shape (B, L, D) where:
             B = batch size
             L = sequence length
             D = number of channels/features
             Can be a tensor or a tuple of tensors
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    result = create_full_like(arr, identity_element)
    
    # Get shape information
    shape = shape_of(arr)
    seq_len = shape[dim]
    
    # Create appropriate indexing for the scan dimension
    for i in range(1, seq_len):
        # Create slices for the current and previous positions
        curr_slice = [slice(None)] * len(shape)
        prev_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        prev_slice[dim] = i-1
        
        # Apply the operation using custom indexing
        set_element(
            result, tuple(curr_slice), 
            op(get_element(result, tuple(prev_slice)), get_element(arr, tuple(prev_slice)))
        )
        
    return result

def blelloch_scan_batched(arr: Any, op: Callable = operator.add, 
                         identity_element: Any = 0, dim: int = 1) -> Any:
    """
    Blelloch parallel exclusive scan implementation supporting batched data, multiple channels,
    and custom data structures with custom indexing.
    
    Args:
        arr: Input data with shape (B, L, D) where:
             B = batch size
             L = sequence length
             D = number of channels/features
             Can be a tensor or a tuple of tensors
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Clone the input to avoid modifying it
    arr = clone_obj(arr)
    
    # Get shape information
    orig_shape = shape_of(arr)
    seq_len = orig_shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr
    
    # Reshape to make the scan dimension the last dimension for easier processing
    # This simplifies the indexing operations
    perm = list(range(len(orig_shape)))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    arr = permute_obj(arr, perm)
    
    # Get new shape after permutation
    shape = shape_of(arr)
    seq_len = shape[-1]
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the sequence dimension if needed
    original_seq_len = seq_len
    if seq_len < pow2:
        # Determine the device for creating paddings
        device = arr[0].device if isinstance(arr, tuple) else arr.device
        dtype = arr[0].dtype if isinstance(arr, tuple) else arr.dtype
        
        # Create padding shape and padding tensor
        padding_shape = list(shape[:-1]) + [pow2 - seq_len]
        
        if isinstance(arr, tuple):
            # For tuple of tensors
            padding_elements = tuple(torch.full(padding_shape, identity_element, device=t.device, dtype=t.dtype) for t in arr)
            arr = cat_obj(arr, padding_elements, dim=-1)
        else:
            # For regular tensors
            padding = torch.full(padding_shape, identity_element, device=device, dtype=dtype)
            arr = torch.cat((arr, padding), dim=-1)
            
        seq_len = pow2
    
    # Combine all batch dimensions for parallel processing
    flat_shape = (-1, seq_len)
    original_shape = shape_of(arr)
    arr = reshape_obj(arr, flat_shape)
    
    # Create a device-specific torch.arange
    if isinstance(arr, tuple):
        device = arr[0].device
    else:
        device = arr.device
    
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
            
            # Update values using custom indexing
            for i in range(len(left_indices)):
                li, ri = left_indices[i].item(), right_indices[i].item()
                left_val = get_element(arr, (slice(None), li))
                right_val = get_element(arr, (slice(None), ri))
                set_element(arr, (slice(None), ri), op(right_val, left_val))
    
    # Set the last element to identity element (for exclusive scan)
    set_element(arr, (slice(None), -1), identity_element)
    
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
            
            # Update values using custom indexing
            for i in range(len(left_indices)):
                li, ri = left_indices[i].item(), right_indices[i].item()
                left_val = get_element(arr, (slice(None), li))
                right_val = get_element(arr, (slice(None), ri))
                
                # Temporary value to avoid in-place modification issues
                temp = clone_obj(left_val)
                set_element(arr, (slice(None), li), right_val)
                set_element(arr, (slice(None), ri), op(right_val, temp))
    
    # Reshape back to original shape (excluding padding)
    original_shape_list = list(original_shape)
    original_shape_list[-1] = original_seq_len
    arr = reshape_obj(arr, tuple(original_shape_list))
    
    # Permute back to original dimension order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    arr = permute_obj(arr, inv_perm)
    
    return arr

def get_identity_element(op: Callable) -> Union[int, float, Tuple]:
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

def verify_scan_batched_tensor(batch_size: int, seq_len: int, channels: int, op: Callable = operator.add, 
                              identity_element: Optional[Union[int, float]] = None, 
                              device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan using regular tensors.
    
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
    
    print(f"\nTesting with standard tensor:")
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

def verify_scan_batched_tuple(batch_size: int, seq_len: int, channels: int, num_tensors: int = 2, 
                             op: Callable = operator.add, identity_element: Optional[Union[int, float]] = None, 
                             device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan using tuple of tensors.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        num_tensors: Number of tensors in the tuple
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create a tuple of random tensors with shape (batch_size, seq_len, channels)
    tensor_tuple = tuple(torch.rand(batch_size, seq_len, channels, device=device) for _ in range(num_tensors))
    
    # Ensure values are small integers for better numerical stability in tests
    tensor_tuple = tuple((t * 10).int().float() for t in tensor_tuple)
    
    # Define custom operation for tuples
    def tuple_op(a, b):
        return tuple(op(a_i, b_i) for a_i, b_i in zip(a, b))
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Run both implementations
    naive_result = naive_scan_batched(tensor_tuple, tuple_op, identity_element)
    blelloch_result = blelloch_scan_batched(tensor_tuple, tuple_op, identity_element)
    
    print(f"\nTesting with tuple of {num_tensors} tensors:")
    print(f"Input tensor shape: {tensor_tuple[0].shape} (each tensor in tuple)")
    print(f"Operator: {op.__name__} (applied element-wise to tuple)")
    print(f"Identity element: {identity_element}")
    
    # Check shapes
    print(f"Naive result first tensor shape: {naive_result[0].shape}")
    print(f"Blelloch result first tensor shape: {blelloch_result[0].shape}")
    
    # Check if results are equal for all tensors in the tuple
    all_equal = True
    for i, (naive_tensor, blelloch_tensor) in enumerate(zip(naive_result, blelloch_result)):
        if not torch.allclose(naive_tensor, blelloch_tensor, rtol=1e-5, atol=1e-5):
            all_equal = False
            
            # Find where differences occur
            diff = (naive_tensor - blelloch_tensor).abs()
            max_diff = diff.max().item()
            max_diff_indices = torch.where(diff == max_diff)
            
            print(f"✗ Verification FAILED for tensor {i}: Max difference: {max_diff}")
            print(f"Max difference at indices: {max_diff_indices}")
            
            # Print sample values where the difference is largest
            b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
            print(f"At batch={b}, pos={l}, channel={c}:")
            print(f"  Naive result: {naive_tensor[b, l, c].item()}")
            print(f"  Blelloch result: {blelloch_tensor[b, l, c].item()}")
    
    if all_equal:
        print("✓ Verification PASSED: Both implementations produce identical results for all tensors in tuple.")
        
        # Print sample results for first batch, first channel of first tensor
        print("\nSample results for first batch, first channel of first tensor:")
        print(f"Input: {tensor_tuple[0][0, :, 0].cpu().numpy()}")
        print(f"Naive scan: {naive_result[0][0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan: {blelloch_result[0][0, :, 0].cpu().numpy()}")
        
        return True
    else:
        return False

def benchmark_scan_batched(batch_size: int, seq_len: int, channels: int, use_tuple: bool = False, 
                          op: Callable = operator.add, identity_element: Optional[Union[int, float]] = None, 
                          iterations: int = 10, device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the batched scan implementations.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        use_tuple: Whether to use a tuple of tensors or a single tensor
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    if use_tuple:
        # Create a tuple of random tensors
        tensor_arr = tuple(torch.rand(batch_size, seq_len, channels, device=device) for _ in range(2))
        
        # Define custom operation for tuples
        def tuple_op(a, b):
            return tuple(op(a_i, b_i) for a_i, b_i in zip(a, b))
        
        operation = tuple_op
    else:
        # Create a single random tensor
        tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
        operation = op
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Warmup
    for _ in range(5):
        naive_scan_batched(tensor_arr, operation, identity_element)
        blelloch_scan_batched(tensor_arr, operation, identity_element)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_batched(tensor_arr, operation, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_batched(tensor_arr, operation, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    input_type = "tuple of tensors" if use_tuple else "standard tensor"
    print(f"\nBenchmark results for shape ({batch_size}, {seq_len}, {channels}) on {device.upper()} using {input_type}:")
    print(f"Operator: {op.__name__}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different batch sizes, sequence lengths, and channel dimensions
print("\n==== Testing and Verification with Standard Tensors ====")

test_configs = [
    (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
    (4, 16, 5),   # Medium config: 4 batches, 16 length, 5 channels
    (8, 32, 2)    # Larger config: 8 batches, 32 length, 2 channels
]

for batch, length, channels in test_configs:
    print(f"\n=== Testing batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched_tensor(batch, length, channels, operator.add, device=device)

# Test with tuple of tensors
print("\n==== Testing and Verification with Tuples of Tensors ====")

for batch, length, channels in test_configs:
    print(f"\n=== Testing tuple of tensors: batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched_tuple(batch, length, channels, num_tensors=2, op=operator.add, device=device)

# Benchmark with different configurations
print("\n==== Performance Benchmarks ====")

benchmark_configs = [
    (1, 100, 1),    # Single batch, medium sequence, single channel
    (10, 100, 10),   # Multiple batches, medium sequence, multiple channels
    (32, 128, 64),   # Typical ML batch/sequence/feature configuration
]

operators = [
    (operator.add, "Addition"),
    (operator.mul, "Multiplication"),
]

for batch, length, channels in benchmark_configs:
    print(f"\n--- Benchmarking batch={batch}, seq_len={length}, channels={channels} ---")
    
    # Benchmark with standard tensor
    for op, name in operators:
        identity = get_identity_element(op)
        print(f"\nOperator: {name} (standard tensor)")
        benchmark_scan_batched(batch, length, channels, use_tuple=False, op=op, identity=identity, device=device)
    
    # Benchmark with tuple of tensors
    for op, name in operators:
        identity = get_identity_element(op)
        print(f"\nOperator: {name} (tuple of tensors)")
        benchmark_scan_batched(batch, length, channels, use_tuple=True, op=op, identity=identity, device=device)