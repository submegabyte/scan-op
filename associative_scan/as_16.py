import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

T = TypeVar('T')  # Type variable for the indexed elements

def naive_scan_batched(arr: T, op: Callable = operator.add, 
                      identity_element: Union[int, float, Tuple] = 0, dim: int = 1,
                      index_fn: Callable[[T, Tuple], Any] = lambda x, idx: x[idx],
                      index_setter: Callable[[T, Tuple, Any], None] = lambda x, idx, val: x.__setitem__(idx, val)) -> T:
    """
    Naive sequential exclusive scan implementation supporting batched data with custom indexing.
    
    Args:
        arr: Input data structure, can be tensor or custom data structure like tuple of tensors
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        index_fn: Function to extract value at a given index (default: standard indexing)
        index_setter: Function to set value at a given index (default: standard item setter)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Clone input if it's a tensor
    if isinstance(arr, torch.Tensor):
        result = torch.full_like(arr, identity_element)
    else:
        # For other data structures, we need to create a new instance and initialize
        # with the identity element. We'll assume arr has a shape attribute.
        result = arr  # This should be a clone or new instance initialized with identity_element
    
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
        
        # Apply the operation using custom indexing
        curr_val = index_fn(result, tuple(prev_slice))
        prev_val = index_fn(arr, tuple(prev_slice))
        new_val = op(curr_val, prev_val)
        index_setter(result, tuple(curr_slice), new_val)
        
    return result

def blelloch_scan_batched(arr: T, op: Callable = operator.add, 
                         identity_element: Union[int, float, Tuple] = 0, dim: int = 1,
                         index_fn: Callable[[T, Tuple], Any] = lambda x, idx: x[idx],
                         index_setter: Callable[[T, Tuple, Any], None] = lambda x, idx, val: x.__setitem__(idx, val)) -> T:
    """
    Blelloch parallel exclusive scan implementation supporting batched data with custom indexing.
    
    Args:
        arr: Input data structure, can be tensor or custom data structure like tuple of tensors
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        index_fn: Function to extract value at a given index (default: standard indexing)
        index_setter: Function to set value at a given index (default: standard item setter)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Clone input if it's a tensor
    if isinstance(arr, torch.Tensor):
        arr = arr.clone()
    else:
        # For other data structures, we'll assume arr has a method to clone itself
        arr = arr.clone() if hasattr(arr, 'clone') else arr  # This should be a clone
    
    # Get shape information
    orig_shape = arr.shape
    seq_len = orig_shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr
    
    # For tensors, we can permute to make operations easier
    # For custom types, we'll need to create a new permute function or work with the original dimension
    if isinstance(arr, torch.Tensor):
        # Reshape to make the scan dimension the last dimension for easier processing
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
        
        # Create a function to access and set elements along the flattened dimension
        def flat_index_fn(data, batch_idx, seq_idx):
            return index_fn(data, (batch_idx, seq_idx))
            
        def flat_index_setter(data, batch_idx, seq_idx, val):
            index_setter(data, (batch_idx, seq_idx), val)
    else:
        # For custom types, we'll work with the original dimensions
        # We still need to handle padding and power of 2 adjustment
        pow2 = 1
        while pow2 < seq_len:
            pow2 *= 2
            
        original_seq_len = seq_len
        # Here we would need to pad the custom data structure if needed
        # This is implementation-specific for each custom data type
        
        # For simplicity in the non-tensor case, we'll use the original indexing
        def flat_index_fn(data, batch_idx, seq_idx):
            # Create full index tuple
            idx = [slice(None)] * len(orig_shape)
            idx[0] = batch_idx  # Assuming batch is first dimension
            idx[dim] = seq_idx
            return index_fn(data, tuple(idx))
            
        def flat_index_setter(data, batch_idx, seq_idx, val):
            idx = [slice(None)] * len(orig_shape)
            idx[0] = batch_idx
            idx[dim] = seq_idx
            index_setter(data, tuple(idx), val)
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = torch.arange(0, seq_len, step, device='cuda' if isinstance(arr, torch.Tensor) and arr.is_cuda else 'cpu')
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values
            # For each batch element:
            if isinstance(arr, torch.Tensor):
                num_batches = arr.shape[0]
            else:
                # Assuming first dimension is batch
                num_batches = orig_shape[0]
                
            for b in range(num_batches):
                for l_idx, r_idx in zip(left_indices.tolist(), right_indices.tolist()):
                    left_val = flat_index_fn(arr, b, l_idx)
                    right_val = flat_index_fn(arr, b, r_idx)
                    new_val = op(right_val, left_val)
                    flat_index_setter(arr, b, r_idx, new_val)
    
    # Set the last element to identity element (for exclusive scan)
    if isinstance(arr, torch.Tensor):
        num_batches = arr.shape[0]
    else:
        num_batches = orig_shape[0]
        
    for b in range(num_batches):
        flat_index_setter(arr, b, seq_len-1, identity_element)
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len))-1, -1, -1):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = torch.arange(0, seq_len, step, device='cuda' if isinstance(arr, torch.Tensor) and arr.is_cuda else 'cpu')
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values
            for b in range(num_batches):
                for l_idx, r_idx in zip(left_indices.tolist(), right_indices.tolist()):
                    left_val = flat_index_fn(arr, b, l_idx)
                    right_val = flat_index_fn(arr, b, r_idx)
                    
                    # Store temporary value to avoid overwriting
                    temp = left_val
                    flat_index_setter(arr, b, l_idx, right_val)
                    flat_index_setter(arr, b, r_idx, op(right_val, temp))
    
    # If tensor, reshape back to original shape
    if isinstance(arr, torch.Tensor):
        # Reshape back to original shape (excluding padding)
        arr = arr.reshape(original_shape)[..., :original_seq_len]
        
        # Permute back to original dimension order
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        arr = arr.permute(*inv_perm)
    else:
        # For custom types, we might need to remove padding here
        # Implementation would be specific to the custom data type
        pass
    
    return arr

def get_identity_element(op: Callable, data_type: Any = None) -> Union[int, float, Tuple]:
    """
    Returns the identity element for common operators and data types.
    
    Args:
        op: The operator function
        data_type: The type of data being operated on (to handle tensor tuples)
        
    Returns:
        The identity element for the operator and data type
    """
    if op is operator.add:
        if data_type is tuple:
            # For tuples, return a tuple of zeros with the same length
            return (0,) * len(data_type(0, 0))  # Assuming data_type is a function returning tuples
        else:
            return 0
    elif op is operator.mul:
        if data_type is tuple:
            # For tuples, return a tuple of ones with the same length
            return (1,) * len(data_type(0, 0))
        else:
            return 1
    elif op is operator.and_:
        if data_type is tuple:
            # For tuples, return a tuple of ones with the same length
            return (1,) * len(data_type(0, 0))
        else:
            return 1  # For bitwise AND, identity is all 1s
    elif op is operator.or_:
        if data_type is tuple:
            # For tuples, return a tuple of zeros with the same length
            return (0,) * len(data_type(0, 0))
        else:
            return 0  # For bitwise OR, identity is all 0s
    else:
        raise ValueError("Unknown operator. Please provide an identity element.")

# Define tuple-specific operations
def tuple_add(a, b):
    """Add two tuples element-wise."""
    return tuple(x + y for x, y in zip(a, b))

def tuple_mul(a, b):
    """Multiply two tuples element-wise."""
    return tuple(x * y for x, y in zip(a, b))

# Custom indexing functions for tensor tuples
def tensor_tuple_index(data, idx):
    """Index a tuple of tensors."""
    return tuple(tensor[idx] for tensor in data)

def tensor_tuple_setter(data, idx, val):
    """Set values in a tuple of tensors."""
    for i, tensor in enumerate(data):
        tensor[idx] = val[i]

class TensorTuple:
    """A wrapper class for a tuple of tensors with the same shape."""
    
    def __init__(self, tensors):
        """
        Initialize with a list or tuple of tensors.
        
        Args:
            tensors: List or tuple of tensors with the same shape
        """
        self.tensors = tuple(tensors)
        self.shape = self.tensors[0].shape
        
    def clone(self):
        """Create a deep copy of this TensorTuple."""
        return TensorTuple([t.clone() for t in self.tensors])
    
    def __getitem__(self, idx):
        """Get values at the specified index from all tensors."""
        return tuple(t[idx] for t in self.tensors)
    
    def __setitem__(self, idx, val):
        """Set values at the specified index for all tensors."""
        for i, t in enumerate(self.tensors):
            t[idx] = val[i]

def verify_scan_batched_tensors(batch_size: int, seq_len: int, channels: int, op: Callable = operator.add, 
                              identity_element: Optional[Union[int, float, Tuple]] = None, 
                              device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan for regular tensors.
    
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

def verify_scan_batched_tuple(batch_size: int, seq_len: int, tuple_size: int, op: Callable = tuple_add, 
                            identity_element: Optional[Tuple] = None, 
                            device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan for tensor tuples.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        tuple_size: Number of elements in each tuple
        op: Binary associative operator (default: tuple_add)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create random tensors for each component in the tuple
    tensors = [torch.rand(batch_size, seq_len, device=device) for _ in range(tuple_size)]
    
    # Ensure values are small integers for better numerical stability in tests
    tensors = [(t * 10).int().float() for t in tensors]
    
    # Create a TensorTuple
    tensor_tuple = TensorTuple(tensors)
    
    # Determine identity element if not provided
    if identity_element is None:
        if op is tuple_add:
            identity_element = tuple(0.0 for _ in range(tuple_size))
        elif op is tuple_mul:
            identity_element = tuple(1.0 for _ in range(tuple_size))
        else:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Custom indexing functions for TensorTuple
    def tuple_index_fn(data, idx):
        return data[idx]
        
    def tuple_index_setter(data, idx, val):
        data[idx] = val
    
    # Run both implementations
    naive_result = naive_scan_batched(
        tensor_tuple.clone(), 
        op, 
        identity_element, 
        dim=1,
        index_fn=tuple_index_fn,
        index_setter=tuple_index_setter
    )
    
    blelloch_result = blelloch_scan_batched(
        tensor_tuple.clone(), 
        op, 
        identity_element, 
        dim=1,
        index_fn=tuple_index_fn,
        index_setter=tuple_index_setter
    )
    
    print(f"Input tensor tuple size: {tuple_size} tensors of shape {tensors[0].shape}")
    print(f"Operator: {op.__name__}")
    
    # Check if results are equal
    all_equal = True
    for i in range(tuple_size):
        if not torch.allclose(naive_result.tensors[i], blelloch_result.tensors[i], rtol=1e-5, atol=1e-5):
            all_equal = False
            break
    
    if all_equal:
        print("✓ Verification PASSED: Both implementations produce identical results.")
        
        # Print sample results for first batch, first component
        print("\nSample results for first batch, first component:")
        print(f"Input: {tensors[0][0, :].cpu().numpy()}")
        print(f"Naive scan: {naive_result.tensors[0][0, :].cpu().numpy()}")
        print(f"Blelloch scan: {blelloch_result.tensors[0][0, :].cpu().numpy()}")
        
        return True
    else:
        # Find where differences occur
        max_diff = 0
        max_diff_indices = (0, 0, 0)
        for i in range(tuple_size):
            diff = (naive_result.tensors[i] - blelloch_result.tensors[i]).abs()
            current_max = diff.max().item()
            if current_max > max_diff:
                max_diff = current_max
                indices = torch.where(diff == current_max)
                max_diff_indices = (i, indices[0][0], indices[1][0])
        
        print(f"✗ Verification FAILED: Max difference: {max_diff}")
        print(f"Max difference at component={max_diff_indices[0]}, batch={max_diff_indices[1]}, pos={max_diff_indices[2]}")
        
        # Print sample values where the difference is largest
        c, b, l = max_diff_indices
        print(f"At component={c}, batch={b}, pos={l}:")
        print(f"  Naive result: {naive_result.tensors[c][b, l].item()}")
        print(f"  Blelloch result: {blelloch_result.tensors[c][b, l].item()}")
        
        return False

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test standard tensors
print("\n==== Testing and Verification with Standard Tensors ====")

test_configs_tensors = [
    (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
    (4, 16, 5),   # Medium config: 4 batches, 16 length, 5 channels
]

for batch, length, channels in test_configs_tensors:
    print(f"\n=== Testing batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched_tensors(batch, length, channels, operator.add, device=device)

# Test tensor tuples
print("\n==== Testing and Verification with Tensor Tuples ====")

test_configs_tuples = [
    (2, 8, 2),    # Small config: 2 batches, 8 length, 2-tuple
    (4, 16, 3),   # Medium config: 4 batches, 16 length, 3-tuple
]

for batch, length, tuple_size in test_configs_tuples:
    print(f"\n=== Testing batch={batch}, seq_len={length}, tuple_size={tuple_size} ===")
    verify_scan_batched_tuple(batch, length, tuple_size, tuple_add, device=device)
    print(f"\n=== Testing with multiplication, batch={batch}, seq_len={length}, tuple_size={tuple_size} ===")
    verify_scan_batched_tuple(batch, length, tuple_size, tuple_mul, device=device)