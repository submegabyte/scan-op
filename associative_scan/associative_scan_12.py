import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any, TypeVar

T = TypeVar('T')

def naive_scan_batched(arr: T, op: Callable = operator.add, 
                      identity_element: Union[int, float] = 0, dim: int = 1,
                      get_item: Callable[[T, tuple], Any] = None,
                      set_item: Callable[[T, tuple, Any], None] = None) -> T:
    """
    Naive sequential exclusive scan implementation supporting batched data, multiple channels,
    and custom data structures through custom indexing operators.
    
    Args:
        arr: Input data structure (tensor, tuple of tensors, etc.)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        get_item: Function to get item at specified indices (default: tensor indexing)
        set_item: Function to set item at specified indices (default: tensor indexing)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Set default indexing operators for tensors if not provided
    if get_item is None:
        get_item = lambda arr, idx: arr[idx]
    if set_item is None:
        def default_set_item(arr, idx, val):
            arr[idx] = val
            return arr
        set_item = default_set_item
    
    # For tensors, create a result tensor filled with identity element
    if isinstance(arr, torch.Tensor):
        result = torch.full_like(arr, identity_element)
        shape = arr.shape
    else:
        # For custom data structures, create a copy of the input
        # We assume the custom data structure provides a way to create a copy
        # filled with the identity element
        result = arr
        # Get shape from the first element if it's a tuple of tensors
        if isinstance(arr, tuple) and all(isinstance(t, torch.Tensor) for t in arr):
            shape = arr[0].shape
        else:
            # This should be handled by the caller for completely custom data structures
            raise ValueError("For non-tensor inputs, shape must be inferrable or custom indexers must handle shape-related operations")
    
    seq_len = shape[dim]
    
    # Create appropriate indexing for the scan dimension
    for i in range(1, seq_len):
        # Create slices for the current and previous positions
        curr_slice = [slice(None)] * len(shape)
        prev_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        prev_slice[dim] = i-1
        
        # Apply the operation using custom indexing
        curr_val = op(get_item(result, tuple(prev_slice)), 
                      get_item(arr, tuple(prev_slice)))
        result = set_item(result, tuple(curr_slice), curr_val)
        
    return result

def blelloch_scan_batched(arr: T, op: Callable = operator.add, 
                         identity_element: Union[int, float] = 0, dim: int = 1,
                         get_item: Callable[[T, Union[tuple, int]], Any] = None,
                         set_item: Callable[[T, Union[tuple, int], Any], T] = None,
                         permute: Callable[[T, List[int]], T] = None,
                         reshape: Callable[[T, tuple], T] = None,
                         get_shape: Callable[[T], tuple] = None,
                         clone: Callable[[T], T] = None,
                         cat: Callable[[List[T], int], T] = None) -> T:
    """
    Blelloch parallel exclusive scan implementation supporting batched data, multiple channels,
    and custom data structures through custom indexing operators.
    
    Args:
        arr: Input data structure (tensor, tuple of tensors, etc.)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        get_item: Function to get item at specified indices
        set_item: Function to set item at specified indices
        permute: Function to permute dimensions
        reshape: Function to reshape the data structure
        get_shape: Function to get the shape of the data structure
        clone: Function to clone the data structure
        cat: Function to concatenate along a dimension
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Set default operators for tensor operations if not provided
    if get_item is None:
        get_item = lambda arr, idx: arr[idx] if isinstance(idx, tuple) else arr[:, idx]
    if set_item is None:
        def default_set_item(arr, idx, val):
            if isinstance(idx, tuple):
                arr[idx] = val
            else:
                arr[:, idx] = val
            return arr
        set_item = default_set_item
    if permute is None:
        permute = lambda arr, dims: arr.permute(*dims)
    if reshape is None:
        reshape = lambda arr, shape: arr.reshape(shape)
    if get_shape is None:
        get_shape = lambda arr: arr.shape
    if clone is None:
        clone = lambda arr: arr.clone()
    if cat is None:
        cat = lambda arrs, dim: torch.cat(arrs, dim=dim)
    
    # Clone the input to avoid modifying it
    arr = clone(arr)
    
    # Get shape information
    orig_shape = get_shape(arr)
    seq_len = orig_shape[dim]
    
    # Handle empty input
    if seq_len == 0:
        return arr
    
    # Reshape to make the scan dimension the last dimension for easier processing
    perm = list(range(len(orig_shape)))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    arr = permute(arr, perm)
    
    # Get new shape after permutation
    shape = get_shape(arr)
    seq_len = shape[-1]
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < seq_len:
        pow2 *= 2
    
    # Pad the sequence dimension if needed
    original_seq_len = seq_len
    if seq_len < pow2:
        if isinstance(arr, torch.Tensor):
            padding_shape = list(shape[:-1]) + [pow2 - seq_len]
            padding = torch.full(padding_shape, identity_element, device=arr.device, dtype=arr.dtype)
            arr = cat([arr, padding], -1)
        else:
            # For custom data structures, the caller must provide a way to pad
            # This might involve creating a padding data structure and concatenating
            raise ValueError("For non-tensor inputs, padding must be handled by custom operators")
        seq_len = pow2
    
    # Combine all batch dimensions for parallel processing
    flat_shape = (-1, seq_len)
    original_shape = get_shape(arr)
    arr = reshape(arr, flat_shape)
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = torch.arange(0, seq_len, step, device="cuda" if isinstance(arr, torch.Tensor) and arr.is_cuda else "cpu")
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values using the custom get/set item operators
            for li, ri in zip(left_indices.tolist(), right_indices.tolist()):
                left_val = get_item(arr, li)
                right_val = get_item(arr, ri)
                arr = set_item(arr, ri, op(right_val, left_val))
    
    # Set the last element to identity element (for exclusive scan)
    arr = set_item(arr, seq_len-1, identity_element)
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len))-1, -1, -1):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = torch.arange(0, seq_len, step, device="cuda" if isinstance(arr, torch.Tensor) and arr.is_cuda else "cpu")
        if indices.numel() > 0:
            left_indices = indices + step//2 - 1
            right_indices = indices + step - 1
            
            # Ensure indices are within bounds
            mask = right_indices < seq_len
            left_indices = left_indices[mask]
            right_indices = right_indices[mask]
            
            # Update values using custom operators
            for li, ri in zip(left_indices.tolist(), right_indices.tolist()):
                left_val = get_item(arr, li)
                right_val = get_item(arr, ri)
                temp = left_val  # Store temporarily to avoid overwriting
                arr = set_item(arr, li, right_val)
                arr = set_item(arr, ri, op(right_val, temp))
    
    # Reshape back to original shape (excluding padding)
    arr = reshape(arr, original_shape)
    
    # For tensors, slice to remove padding
    if isinstance(arr, torch.Tensor):
        # Create slices to select all elements except for padding
        slices = [slice(None) for _ in range(len(original_shape)-1)] + [slice(0, original_seq_len)]
        arr = arr[tuple(slices)]
    else:
        # For custom data structures, the caller must provide a way to handle this
        pass
    
    # Permute back to original dimension order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    arr = permute(arr, inv_perm)
    
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

class TensorPair:
    """
    Example wrapper class for pairs of tensors to demonstrate custom indexing.
    """
    def __init__(self, tensor1, tensor2, fill_value=0):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.fill_value = fill_value
    
    @property
    def shape(self):
        return self.tensor1.shape
    
    @property
    def device(self):
        return self.tensor1.device
    
    def clone(self):
        return TensorPair(self.tensor1.clone(), self.tensor2.clone(), self.fill_value)
    
    def __repr__(self):
        return f"TensorPair(tensor1={self.tensor1}, tensor2={self.tensor2})"

def tensor_pair_get_item(pair, idx):
    """Custom get_item function for TensorPair."""
    return TensorPair(pair.tensor1[idx], pair.tensor2[idx], pair.fill_value)

def tensor_pair_set_item(pair, idx, val):
    """Custom set_item function for TensorPair."""
    result = pair.clone()
    result.tensor1[idx] = val.tensor1
    result.tensor2[idx] = val.tensor2
    return result

def tensor_pair_permute(pair, dims):
    """Custom permute function for TensorPair."""
    return TensorPair(pair.tensor1.permute(*dims), pair.tensor2.permute(*dims), pair.fill_value)

def tensor_pair_reshape(pair, shape):
    """Custom reshape function for TensorPair."""
    return TensorPair(pair.tensor1.reshape(shape), pair.tensor2.reshape(shape), pair.fill_value)

def tensor_pair_get_shape(pair):
    """Custom get_shape function for TensorPair."""
    return pair.shape

def tensor_pair_clone(pair):
    """Custom clone function for TensorPair."""
    return pair.clone()

def tensor_pair_cat(pairs, dim):
    """Custom concatenation function for TensorPair."""
    tensors1 = [p.tensor1 for p in pairs]
    tensors2 = [p.tensor2 for p in pairs]
    return TensorPair(torch.cat(tensors1, dim=dim), torch.cat(tensors2, dim=dim), pairs[0].fill_value)

def tensor_pair_op_add(a, b):
    """Custom addition operator for TensorPair."""
    return TensorPair(a.tensor1 + b.tensor1, a.tensor2 + b.tensor2, a.fill_value)

def tensor_pair_op_mul(a, b):
    """Custom multiplication operator for TensorPair."""
    return TensorPair(a.tensor1 * b.tensor1, a.tensor2 * b.tensor2, a.fill_value)

def verify_scan_batched_with_custom_indexing(batch_size: int, seq_len: int, channels: int, 
                                            op: Callable = operator.add, 
                                            custom_data_type: str = "tensor",
                                            identity_element: Optional[Union[int, float]] = None, 
                                            device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan with custom indexing.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        op: Binary associative operator (default: addition)
        custom_data_type: Type of data structure to use ("tensor" or "tensor_pair")
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    if custom_data_type == "tensor":
        # Create random tensor with shape (batch_size, seq_len, channels)
        data = torch.rand(batch_size, seq_len, channels, device=device)
        # Ensure values are small integers for better numerical stability in tests
        data = (data * 10).int().float()
        
        # Define custom indexing operators (standard tensor operations)
        get_item = lambda arr, idx: arr[idx]
        set_item = lambda arr, idx, val: arr.clone().index_put_((idx,), val)
        
        # Run both implementations
        naive_result = naive_scan_batched(data.clone(), op, identity_element, 
                                         get_item=get_item, set_item=set_item)
        blelloch_result = blelloch_scan_batched(data.clone(), op, identity_element)
        
    elif custom_data_type == "tensor_pair":
        # Create a pair of random tensors
        tensor1 = torch.rand(batch_size, seq_len, channels, device=device) * 10
        tensor2 = torch.rand(batch_size, seq_len, channels, device=device) * 10
        data = TensorPair(tensor1, tensor2, fill_value=identity_element)
        
        # Define custom operators for TensorPair
        if op is operator.add:
            pair_op = tensor_pair_op_add
        elif op is operator.mul:
            pair_op = tensor_pair_op_mul
        else:
            raise ValueError(f"Operator {op.__name__} not supported for TensorPair")
        
        # Run both implementations with custom operators
        naive_result = naive_scan_batched(
            data.clone(), pair_op, TensorPair(torch.full_like(tensor1, identity_element), 
                                             torch.full_like(tensor2, identity_element), 
                                             identity_element),
            get_item=tensor_pair_get_item, 
            set_item=tensor_pair_set_item
        )
        
        blelloch_result = blelloch_scan_batched(
            data.clone(), pair_op, TensorPair(torch.full_like(tensor1, identity_element), 
                                             torch.full_like(tensor2, identity_element), 
                                             identity_element),
            get_item=tensor_pair_get_item, 
            set_item=tensor_pair_set_item,
            permute=tensor_pair_permute,
            reshape=tensor_pair_reshape,
            get_shape=tensor_pair_get_shape,
            clone=tensor_pair_clone,
            cat=tensor_pair_cat
        )
    else:
        raise ValueError(f"Unsupported custom data type: {custom_data_type}")
    
    print(f"Input data type: {custom_data_type}")
    if custom_data_type == "tensor":
        print(f"Input tensor shape: {data.shape}")
    else:
        print(f"Input tensor pair shapes: {data.tensor1.shape}, {data.tensor2.shape}")
    
    print(f"Operator: {op.__name__}")
    print(f"Identity element: {identity_element}")
    
    # Check if results are equal
    if custom_data_type == "tensor":
        is_equal = torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5)
        
        if is_equal:
            print("✓ Verification PASSED: Both implementations produce identical results.")
            
            # Print sample results for first batch, first channel
            print("\nSample results for first batch, first channel:")
            print(f"Input: {data[0, :, 0].cpu().numpy()}")
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
    
    elif custom_data_type == "tensor_pair":
        is_equal_tensor1 = torch.allclose(naive_result.tensor1, blelloch_result.tensor1, rtol=1e-5, atol=1e-5)
        is_equal_tensor2 = torch.allclose(naive_result.tensor2, blelloch_result.tensor2, rtol=1e-5, atol=1e-5)
        is_equal = is_equal_tensor1 and is_equal_tensor2
        
        if is_equal:
            print("✓ Verification PASSED: Both implementations produce identical results for tensor pair.")
            
            # Print sample results for first batch, first channel
            print("\nSample results for first batch, first channel (tensor1):")
            print(f"Input: {data.tensor1[0, :, 0].cpu().numpy()}")
            print(f"Naive scan: {naive_result.tensor1[0, :, 0].cpu().numpy()}")
            print(f"Blelloch scan: {blelloch_result.tensor1[0, :, 0].cpu().numpy()}")
            
            print("\nSample results for first batch, first channel (tensor2):")
            print(f"Input: {data.tensor2[0, :, 0].cpu().numpy()}")
            print(f"Naive scan: {naive_result.tensor2[0, :, 0].cpu().numpy()}")
            print(f"Blelloch scan: {blelloch_result.tensor2[0, :, 0].cpu().numpy()}")
            
            return True
        else:
            if not is_equal_tensor1:
                # Find where differences occur in tensor1
                diff = (naive_result.tensor1 - blelloch_result.tensor1).abs()
                max_diff = diff.max().item()
                max_diff_indices = torch.where(diff == max_diff)
                
                print(f"✗ Verification FAILED for tensor1: Max difference: {max_diff}")
                print(f"Max difference at indices: {max_diff_indices}")
                
                # Print sample values where the difference is largest
                b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
                print(f"At batch={b}, pos={l}, channel={c}:")
                print(f"  Naive result: {naive_result.tensor1[b, l, c].item()}")
                print(f"  Blelloch result: {blelloch_result.tensor1[b, l, c].item()}")
            
            if not is_equal_tensor2:
                # Find where differences occur in tensor2
                diff = (naive_result.tensor2 - blelloch_result.tensor2).abs()
                max_diff = diff.max().item()
                max_diff_indices = torch.where(diff == max_diff)
                
                print(f"✗ Verification FAILED for tensor2: Max difference: {max_diff}")
                print(f"Max difference at indices: {max_diff_indices}")
                
                # Print sample values where the difference is largest
                b, l, c = max_diff_indices[0][0], max_diff_indices[1][0], max_diff_indices[2][0]
                print(f"At batch={b}, pos={l}, channel={c}:")
                print(f"  Naive result: {naive_result.tensor2[b, l, c].item()}")
                print(f"  Blelloch result: {blelloch_result.tensor2[b, l, c].item()}")
            
            return False

# Example usage

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with standard tensors
print("\n==== Testing with Standard Tensors ====")
test_configs = [
    (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
    (4, 16, 5),   # Medium config: 4 batches, 16 length, 5 channels
]

for batch, length, channels in test_configs:
    print(f"\n=== Testing with tensors: batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched_with_custom_indexing(batch, length, channels, operator.add, 
                                            custom_data_type="tensor", device=device)

# Test with tensor pairs
print("\n==== Testing with Tensor Pairs ====")
for batch, length, channels in test_configs:
    print(f"\n=== Testing with tensor pairs: batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched_with_custom_indexing(batch, length, channels, operator.add, 
                                            custom_data_type="tensor_pair", device=device)
    
    print(f"\n=== Testing multiplication with tensor pairs: batch={batch}, seq_len={length}, channels={channels} ===")
    verify_scan_batched_with_custom_indexing(batch, length, channels, operator.mul, 
                                            custom_data_type="tensor_pair", device=device)