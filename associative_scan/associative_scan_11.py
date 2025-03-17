import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

T = TypeVar('T')

def naive_scan_batched(arr: T, op: Callable = operator.add, 
                      identity_element: Union[int, float] = 0, dim: int = 1,
                      index_fn: Callable[[T, Tuple], Any] = None,
                      set_fn: Callable[[T, Tuple, Any], None] = None) -> T:
    """
    Naive sequential exclusive scan implementation supporting batched data and multiple channels.
    Now supports custom data structures through index_fn and set_fn.
    
    Args:
        arr: Input data structure (tensor or other indexable structure)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        index_fn: Function to get elements from arr (default: None, uses standard tensor indexing)
        set_fn: Function to set elements in the result (default: None, uses standard tensor assignment)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Default indexing/setting functions for tensors
    if index_fn is None:
        index_fn = lambda x, idx: x[idx]
    if set_fn is None:
        set_fn = lambda x, idx, val: x.__setitem__(idx, val)
    
    # For tensor inputs, create a tensor filled with identity elements
    if isinstance(arr, torch.Tensor):
        result = torch.full_like(arr, identity_element)
    else:
        # For custom data structures, create a copy
        import copy
        result = copy.deepcopy(arr)
        # Initialize with identity element if needed
        # This depends on the specific data structure
    
    # Get shape information
    if isinstance(arr, torch.Tensor):
        shape = arr.shape
        seq_len = shape[dim]
    else:
        # For custom data structures, we need to determine the sequence length
        # This is data structure specific and should be handled by the user
        if hasattr(arr, "shape"):
            shape = arr.shape
            seq_len = shape[dim]
        else:
            raise ValueError("Custom data structure must provide sequence length information")
    
    # Create appropriate indexing for the scan dimension
    for i in range(1, seq_len):
        # Create slices for the current and previous positions
        curr_slice = [slice(None)] * len(shape)
        prev_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        prev_slice[dim] = i-1
        
        # Apply the operation using custom index and set functions
        curr_tuple = tuple(curr_slice)
        prev_tuple = tuple(prev_slice)
        
        # Get values using index function
        prev_result = index_fn(result, prev_tuple)
        prev_input = index_fn(arr, prev_tuple)
        
        # Apply operation
        new_val = op(prev_result, prev_input)
        
        # Set value using set function
        set_fn(result, curr_tuple, new_val)
        
    return result

def blelloch_scan_batched(arr: T, op: Callable = operator.add, 
                         identity_element: Union[int, float] = 0, dim: int = 1,
                         index_fn: Callable[[T, Tuple], Any] = None,
                         set_fn: Callable[[T, Tuple, Any], None] = None) -> T:
    """
    Blelloch parallel exclusive scan implementation supporting batched data and multiple channels.
    Now supports custom data structures through index_fn and set_fn.
    
    Args:
        arr: Input data structure (tensor or other indexable structure)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        index_fn: Function to get elements from arr (default: None, uses standard tensor indexing)
        set_fn: Function to set elements in the result (default: None, uses standard tensor assignment)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Default indexing/setting functions for tensors
    if index_fn is None:
        index_fn = lambda x, idx: x[idx]
    if set_fn is None:
        set_fn = lambda x, idx, val: x.__setitem__(idx, val)
    
    # Handle non-tensor inputs
    if not isinstance(arr, torch.Tensor):
        # Clone or deep copy the input to avoid modifying it
        import copy
        arr_copy = copy.deepcopy(arr)
    else:
        # Clone the input to avoid modifying it
        arr_copy = arr.clone()
    
    # Get shape information
    if isinstance(arr, torch.Tensor):
        orig_shape = arr.shape
        seq_len = orig_shape[dim]
    else:
        # For custom data structures, we need to determine the sequence length
        if hasattr(arr, "shape"):
            orig_shape = arr.shape
            seq_len = orig_shape[dim]
        else:
            raise ValueError("Custom data structure must provide sequence length information")
    
    # Handle empty input
    if seq_len == 0:
        return arr_copy
    
    # For tensor inputs, handle permutation and reshaping
    if isinstance(arr, torch.Tensor):
        # Reshape to make the scan dimension the last dimension for easier processing
        perm = list(range(len(orig_shape)))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        arr_copy = arr_copy.permute(*perm)
        
        # Get new shape after permutation
        shape = arr_copy.shape
        seq_len = shape[-1]
        
        # Round up to the next power of 2
        pow2 = 1
        while pow2 < seq_len:
            pow2 *= 2
        
        # Pad the sequence dimension if needed
        original_seq_len = seq_len
        if seq_len < pow2:
            padding_shape = list(shape[:-1]) + [pow2 - seq_len]
            padding = torch.full(padding_shape, identity_element, device=arr_copy.device, dtype=arr_copy.dtype)
            arr_copy = torch.cat((arr_copy, padding), dim=-1)
            seq_len = pow2
        
        # Combine all batch dimensions for parallel processing
        flat_shape = (-1, seq_len)
        original_shape = arr_copy.shape
        arr_copy = arr_copy.reshape(flat_shape)
        
        # Get device for creating indices
        device = arr_copy.device
    else:
        # For non-tensor inputs, we need custom handling
        # This is a simplified version - in practice would need more structure-specific logic
        original_seq_len = seq_len
        pow2 = 1
        while pow2 < seq_len:
            pow2 *= 2
        
        # For creating indices, use CPU device as default
        device = torch.device("cpu")
    
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
            
            # Update values using custom index and set functions
            for i in range(len(left_indices)):
                li, ri = left_indices[i].item(), right_indices[i].item()
                
                if isinstance(arr, torch.Tensor):
                    # For tensor, use batch operations
                    arr_copy[:, ri] = op(arr_copy[:, ri], arr_copy[:, li])
                else:
                    # For non-tensor, use the provided index and set functions
                    left_idx = tuple([slice(None)] * (len(orig_shape)-1) + [li])
                    right_idx = tuple([slice(None)] * (len(orig_shape)-1) + [ri])
                    
                    right_val = index_fn(arr_copy, right_idx)
                    left_val = index_fn(arr_copy, left_idx)
                    new_val = op(right_val, left_val)
                    set_fn(arr_copy, right_idx, new_val)
    
    # Set the last element to identity element (for exclusive scan)
    if isinstance(arr, torch.Tensor):
        arr_copy[:, -1] = identity_element
    else:
        last_idx = tuple([slice(None)] * (len(orig_shape)-1) + [seq_len-1])
        set_fn(arr_copy, last_idx, identity_element)
    
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
            
            # Update values using custom index and set functions
            for i in range(len(left_indices)):
                li, ri = left_indices[i].item(), right_indices[i].item()
                
                if isinstance(arr, torch.Tensor):
                    # For tensor, use batch operations with a temporary tensor
                    temp = arr_copy[:, li].clone()
                    arr_copy[:, li] = arr_copy[:, ri]
                    arr_copy[:, ri] = op(arr_copy[:, ri], temp)
                else:
                    # For non-tensor, use the provided index and set functions
                    left_idx = tuple([slice(None)] * (len(orig_shape)-1) + [li])
                    right_idx = tuple([slice(None)] * (len(orig_shape)-1) + [ri])
                    
                    left_val = index_fn(arr_copy, left_idx)
                    right_val = index_fn(arr_copy, right_idx)
                    
                    # Temporary store left value
                    temp = left_val
                    
                    # Set left value to right value
                    set_fn(arr_copy, left_idx, right_val)
                    
                    # Update right value
                    new_right_val = op(right_val, temp)
                    set_fn(arr_copy, right_idx, new_right_val)
    
    # For tensor inputs, handle reshaping back
    if isinstance(arr, torch.Tensor):
        # Reshape back to original shape (excluding padding)
        arr_copy = arr_copy.reshape(original_shape)[..., :original_seq_len]
        
        # Permute back to original dimension order
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        arr_copy = arr_copy.permute(*inv_perm)
    
    return arr_copy

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

# Class for testing with tensor tuples
class TensorTuple:
    def __init__(self, tensors):
        """
        Initialize a TensorTuple with a list of tensors.
        All tensors should have the same shape.
        
        Args:
            tensors: List of tensors with same shape
        """
        self.tensors = tensors
        
        # Verify all tensors have the same shape
        if len(tensors) > 0:
            self.shape = tensors[0].shape
            for t in tensors[1:]:
                if t.shape != self.shape:
                    raise ValueError("All tensors in TensorTuple must have the same shape")
        else:
            self.shape = None
    
    def clone(self):
        """Create a deep copy of this TensorTuple"""
        return TensorTuple([t.clone() for t in self.tensors])
    
    def __len__(self):
        return len(self.tensors)

# Define index and set functions for TensorTuple
def tensor_tuple_index_fn(tensor_tuple, idx):
    """
    Index function for TensorTuple.
    
    Args:
        tensor_tuple: TensorTuple instance
        idx: Tuple of indices/slices
        
    Returns:
        New TensorTuple with indexed tensors
    """
    return TensorTuple([t[idx] for t in tensor_tuple.tensors])

def tensor_tuple_set_fn(tensor_tuple, idx, value):
    """
    Set function for TensorTuple.
    
    Args:
        tensor_tuple: TensorTuple instance to modify
        idx: Tuple of indices/slices
        value: TensorTuple with new values
    """
    for i in range(len(tensor_tuple.tensors)):
        tensor_tuple.tensors[i][idx] = value.tensors[i]

# Custom operator for TensorTuple
def tensor_tuple_add(a, b):
    """
    Add two TensorTuples element-wise.
    
    Args:
        a: First TensorTuple
        b: Second TensorTuple
        
    Returns:
        New TensorTuple with element-wise sum
    """
    if isinstance(a, TensorTuple) and isinstance(b, TensorTuple):
        return TensorTuple([a.tensors[i] + b.tensors[i] for i in range(len(a.tensors))])
    else:
        raise TypeError("Both arguments must be TensorTuple")

def verify_scan_batched(batch_size: int, seq_len: int, channels: int, op: Callable = operator.add, 
                       identity_element: Optional[Union[int, float]] = None, 
                       device: str = "cpu", use_tensor_tuple: bool = False) -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        device: "cpu" or "cuda" for GPU (if available)
        use_tensor_tuple: If True, test with TensorTuple instead of regular tensor
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create random tensor or TensorTuple
    if use_tensor_tuple:
        # Create 2 tensors for the tuple
        tensor1 = torch.rand(batch_size, seq_len, channels, device=device)
        tensor2 = torch.rand(batch_size, seq_len, channels, device=device)
        
        # Ensure values are small integers
        tensor1 = (tensor1 * 10).int().float()
        tensor2 = (tensor2 * 10).int().float()
        
        tensor_arr = TensorTuple([tensor1, tensor2])
        
        # Use custom index and set functions for TensorTuple
        index_fn = tensor_tuple_index_fn
        set_fn = tensor_tuple_set_fn
        
        # Use custom TensorTuple operator
        if op is operator.add:
            op = tensor_tuple_add
    else:
        # Create regular tensor
        tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
        tensor_arr = (tensor_arr * 10).int().float()
        
        # Use default index and set functions
        index_fn = None
        set_fn = None
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Create identity element for TensorTuple if needed
    if use_tensor_tuple and not isinstance(identity_element, TensorTuple):
        identity_element = TensorTuple([torch.tensor(identity_element, device=device) for _ in range(len(tensor_arr.tensors))])
    
    # Run both implementations
    naive_result = naive_scan_batched(tensor_arr, op, identity_element, index_fn=index_fn, set_fn=set_fn)
    blelloch_result = blelloch_scan_batched(tensor_arr, op, identity_element, index_fn=index_fn, set_fn=set_fn)
    
    print(f"Input {'TensorTuple' if use_tensor_tuple else 'tensor'} shape: {tensor_arr.shape}")
    print(f"Operator: {op.__name__ if hasattr(op, '__name__') else 'custom'}")
    
    # Check if results are equal
    if use_tensor_tuple:
        # For TensorTuple, check if all tensors are equal
        all_equal = True
        max_diff = 0.0
        for i in range(len(naive_result.tensors)):
            if not torch.allclose(naive_result.tensors[i], blelloch_result.tensors[i], rtol=1e-5, atol=1e-5):
                all_equal = False
                diff = (naive_result.tensors[i] - blelloch_result.tensors[i]).abs()
                max_diff = max(max_diff, diff.max().item())
        
        if all_equal:
            print("✓ Verification PASSED: Both implementations produce identical results.")
            
            # Print sample results for first batch, first channel, first tensor
            print("\nSample results for first batch, first channel, first tensor:")
            print(f"Input: {tensor_arr.tensors[0][0, :, 0].cpu().numpy()}")
            print(f"Naive scan: {naive_result.tensors[0][0, :, 0].cpu().numpy()}")
            print(f"Blelloch scan: {blelloch_result.tensors[0][0, :, 0].cpu().numpy()}")
            
            return True
        else:
            print(f"✗ Verification FAILED: Max difference: {max_diff}")
            return False
    else:
        # For regular tensor, use torch.allclose
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
                          iterations: int = 10, device: str = "cpu",
                          use_tensor_tuple: bool = False) -> Tuple[float, float]:
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
        use_tensor_tuple: If True, test with TensorTuple instead of regular tensor
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Create random tensor or TensorTuple
    if use_tensor_tuple:
        # Create 2 tensors for the tuple
        tensor1 = torch.rand(batch_size, seq_len, channels, device=device)
        tensor2 = torch.rand(batch_size, seq_len, channels, device=device)
        
        tensor_arr = TensorTuple([tensor1, tensor2])
        
        # Use custom index and set functions for TensorTuple
        index_fn = tensor_tuple_index_fn
        set_fn = tensor_tuple_set_fn
        
        # Use custom TensorTuple operator
        if op is operator.add:
            op = tensor_tuple_add
    else:
        # Create regular tensor
        tensor_arr = torch.rand(batch_size, seq_len, channels, device=device)
        
        # Use default index and set functions
        index_fn = None
        set_fn = None
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Create identity element for TensorTuple if needed
    if use_tensor_tuple and not isinstance(identity_element, TensorTuple):
        identity_element = TensorTuple([torch.tensor(identity_element, device=device) for _ in range(len(tensor_arr.tensors))])
    
    # Warmup
    for _ in range(5):
        naive_scan_batched(tensor_arr, op, identity_element, index_fn=index_fn, set_fn=set_fn)
        blelloch_scan_batched(tensor_arr, op, identity_element, index_fn=index_fn, set_fn=set_fn)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_batched(tensor_arr, op, identity_element, index_fn=index_fn, set_fn=set_fn)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_batched(tensor_arr, op, identity_element, index_fn=index_fn, set_fn=set_fn)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results for {'TensorTuple' if use_tensor_tuple else 'tensor'} shape ({batch_size}, {seq_len}, {channels}) on {device.upper()}:")
    print(f"Operator: {op.__name__ if hasattr(op, '__name__') else 'custom'}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different batch sizes, sequence lengths, and channel dimensions
print("\n==== Testing and Verification with Batched Data (Standard Tensors) ====")

test_configs = [
    (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
    (4, 16, 5),   # Medium config: 4 batches, 16 length, 5 channels
    (8, 32, 2)    # Larger config: 8 batches, 32 length, 2 channels
]

for batch, length, channels in test_configs:
    print(f"\n=== Testing batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched(batch, length, channels, operator.add, device=device, use_tensor_tuple=False)

# Test with TensorTuple
print("\n==== Testing and Verification with Batched Data (TensorTuple) ====")

for batch, length, channels in test_configs:
    print(f"\n=== Testing TensorTuple batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched(batch, length, channels, operator.add, device=device, use_tensor_tuple=True)

# Benchmark with different configurations
print("\n==== Performance Benchmarks with Batched Data (Standard Tensors) ====")

benchmark_configs = [
    (1, 1000, 1),     # Single batch, long sequence, single channel
    (10, 100, 1),     # Multiple batches, medium sequence, single channel
    (10, 100, 10),    # Multiple batches, medium sequence, multiple channels
    (32, 128, 64),    # Typical ML batch/sequence/feature configuration
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
        benchmark_scan_batched(batch, length, channels, op, identity, device=device, use_tensor_tuple=False)

# Benchmark with TensorTuple
print("\n==== Performance Benchmarks with Batched Data (TensorTuple) ====")

for batch, length, channels in benchmark_configs[:2]:  # Use smaller configs for TensorTuple to save time
    print(f"\n--- Benchmarking TensorTuple batch={batch}, seq_len={length}, channels={channels} ---")
    
    # Only test with addition for TensorTuple
    print(f"\nOperator: Addition (TensorTuple)")
    benchmark_scan_batched(batch, length, channels, operator.add, 0, device=device, use_tensor_tuple=True)