import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional, Any

def default_indexer(obj, idx):
    """Default indexing function that works with standard tensor indexing"""
    return obj[idx]

def default_setter(obj, idx, value):
    """Default setter function that works with standard tensor assignment"""
    obj[idx] = value
    return obj

def naive_scan_batched(arr: Any, op: Callable = operator.add, 
                      identity_element: Any = 0, dim: int = 1,
                      indexer: Callable = default_indexer,
                      setter: Callable = default_setter) -> Any:
    """
    Naive sequential exclusive scan implementation supporting batched data and multiple channels,
    with custom indexing for non-tensor types.
    
    Args:
        arr: Input data structure (tensor, tuple of tensors, etc.)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        indexer: Custom indexing function (default: standard tensor indexing)
        setter: Custom setter function (default: standard tensor assignment)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Create a result structure filled with identity elements
    result = torch.full_like(arr, identity_element) if isinstance(arr, torch.Tensor) else [identity_element] * len(arr)
    
    # Get shape information from the first element if dealing with a non-tensor
    if isinstance(arr, torch.Tensor):
        shape = arr.shape
    else:
        # For non-tensor types, get shape from the first element
        first_elem = arr[0]
        if isinstance(first_elem, torch.Tensor):
            shape = first_elem.shape
        else:
            # For nested structures, try to determine the length along the scan dimension
            shape = [len(arr)]
    
    seq_len = shape[dim] if dim < len(shape) else len(arr)
    
    # Create appropriate indexing for the scan dimension
    for i in range(1, seq_len):
        # Create slices for the current and previous positions
        curr_slice = [slice(None)] * len(shape)
        prev_slice = [slice(None)] * len(shape)
        curr_slice[dim] = i
        prev_slice[dim] = i-1
        
        # Apply the operation using custom indexing and setting
        curr_tuple = tuple(curr_slice)
        prev_tuple = tuple(prev_slice)
        
        prev_value = indexer(arr, prev_tuple)
        prev_result = indexer(result, prev_tuple)
        
        # Apply the operation and set the result
        new_value = op(prev_result, prev_value)
        result = setter(result, curr_tuple, new_value)
        
    return result

def blelloch_scan_batched(arr: Any, op: Callable = operator.add, 
                         identity_element: Any = 0, dim: int = 1,
                         indexer: Callable = default_indexer,
                         setter: Callable = default_setter) -> Any:
    """
    Blelloch parallel exclusive scan implementation supporting batched data and multiple channels,
    with custom indexing for non-tensor types.
    
    Args:
        arr: Input data structure (tensor, tuple of tensors, etc.)
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1, the sequence dimension)
        indexer: Custom indexing function (default: standard tensor indexing)
        setter: Custom setter function (default: standard tensor assignment)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Clone the input to avoid modifying it
    if isinstance(arr, torch.Tensor):
        arr_copy = arr.clone()
        # Get shape information
        orig_shape = arr_copy.shape
        seq_len = orig_shape[dim]
        
        # Handle empty input
        if seq_len == 0:
            return arr_copy
        
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
        
        # Use standard tensor operations for tensor input
        indexer_func = default_indexer
        setter_func = default_setter
    else:
        # For non-tensor types, create a deep copy if possible, or convert to a suitable format
        try:
            import copy
            arr_copy = copy.deepcopy(arr)
        except:
            arr_copy = list(arr)  # Fallback to list conversion
        
        # Determine sequence length for non-tensor types
        if hasattr(arr, '__len__'):
            seq_len = len(arr)
        else:
            raise ValueError("Cannot determine sequence length for non-sequence input")
        
        # Round up to the next power of 2
        original_seq_len = seq_len
        pow2 = 1
        while pow2 < seq_len:
            pow2 *= 2
        
        # Pad with identity elements if needed
        if seq_len < pow2:
            if isinstance(arr_copy, list):
                arr_copy = arr_copy + [identity_element] * (pow2 - seq_len)
            else:
                # For other sequence types, convert to list for padding
                arr_copy = list(arr_copy) + [identity_element] * (pow2 - seq_len)
            seq_len = pow2
        
        # Use provided custom indexing functions
        indexer_func = indexer
        setter_func = setter
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(seq_len))):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = range(0, seq_len, step)
        
        for idx in indices:
            left_idx = idx + step//2 - 1
            right_idx = idx + step - 1
            
            # Ensure indices are within bounds
            if right_idx < seq_len:
                # Get values using custom indexer
                left_val = indexer_func(arr_copy, (slice(None), left_idx) if isinstance(arr_copy, torch.Tensor) else left_idx)
                right_val = indexer_func(arr_copy, (slice(None), right_idx) if isinstance(arr_copy, torch.Tensor) else right_idx)
                
                # Apply operation
                new_right = op(right_val, left_val)
                
                # Set value using custom setter
                arr_copy = setter_func(arr_copy, (slice(None), right_idx) if isinstance(arr_copy, torch.Tensor) else right_idx, new_right)
    
    # Set the last element to identity element (for exclusive scan)
    arr_copy = setter_func(arr_copy, (slice(None), -1) if isinstance(arr_copy, torch.Tensor) else -1, identity_element)
    
    # Down-sweep phase
    for d in range(int(math.log2(seq_len))-1, -1, -1):
        step = 2 ** (d+1)
        
        # Create indices for the operation
        indices = range(0, seq_len, step)
        
        for idx in indices:
            left_idx = idx + step//2 - 1
            right_idx = idx + step - 1
            
            # Ensure indices are within bounds
            if right_idx < seq_len:
                # Get values using custom indexer
                left_val = indexer_func(arr_copy, (slice(None), left_idx) if isinstance(arr_copy, torch.Tensor) else left_idx)
                right_val = indexer_func(arr_copy, (slice(None), right_idx) if isinstance(arr_copy, torch.Tensor) else right_idx)
                
                # Store left value temporarily
                temp = left_val
                
                # Update left with right
                arr_copy = setter_func(arr_copy, (slice(None), left_idx) if isinstance(arr_copy, torch.Tensor) else left_idx, right_val)
                
                # Update right with operation
                new_right = op(right_val, temp)
                arr_copy = setter_func(arr_copy, (slice(None), right_idx) if isinstance(arr_copy, torch.Tensor) else right_idx, new_right)
    
    # Post-processing for tensor types
    if isinstance(arr_copy, torch.Tensor):
        # Reshape back to original shape (excluding padding)
        arr_copy = arr_copy.reshape(original_shape)[..., :original_seq_len]
        
        # Permute back to original dimension order
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        arr_copy = arr_copy.permute(*inv_perm)
    else:
        # Truncate to original length for non-tensor types
        if hasattr(arr_copy, '__getitem__') and hasattr(arr_copy, '__len__'):
            arr_copy = arr_copy[:original_seq_len]
    
    return arr_copy

def get_identity_element(op: Callable, input_type: Any = None) -> Any:
    """
    Returns the identity element for common operators and input types.
    
    Args:
        op: The operator function
        input_type: Optional sample of the input type to determine proper identity element
        
    Returns:
        The identity element for the operator and input type
    """
    # Handle standard operators with tensor inputs
    if input_type is None or isinstance(input_type, (int, float, torch.Tensor)):
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
    # Handle tuple/list of tensors or other custom types
    elif isinstance(input_type, (tuple, list)):
        if op is operator.add:
            # Return a tuple/list of zeros with same structure as input
            return type(input_type)(0 for _ in input_type)
        elif op is operator.mul:
            # Return a tuple/list of ones with same structure as input
            return type(input_type)(1 for _ in input_type)
        elif op is operator.and_:
            # Return a tuple/list of ones with same structure as input
            return type(input_type)(1 for _ in input_type)
        elif op is operator.or_:
            # Return a tuple/list of zeros with same structure as input
            return type(input_type)(0 for _ in input_type)
        elif op is torch.max:
            # Return a tuple/list of -inf with same structure as input
            return type(input_type)(float('-inf') for _ in input_type)
        elif op is torch.min:
            # Return a tuple/list of inf with same structure as input
            return type(input_type)(float('inf') for _ in input_type)
        
    # Default case for unknown operators or types
    raise ValueError("Unknown operator or input type. Please provide an identity element.")

def tuple_indexer(obj, idx):
    """
    Custom indexer for tuples of tensors.
    Returns a tuple of indexed values.
    """
    if isinstance(idx, tuple):
        # For multi-dimensional indexing (like tensor batches)
        if len(idx) == 2 and isinstance(idx[0], slice) and idx[0] == slice(None):
            # This handles the case (slice(None), i) used in the Blelloch algorithm
            position_idx = idx[1]
            return tuple(t[:, position_idx] if isinstance(t, torch.Tensor) and t.dim() > 1 else t[position_idx] for t in obj)
        else:
            # General multi-dimensional case
            return tuple(t[idx] if isinstance(t, torch.Tensor) else t for t in obj)
    else:
        # For single-dimension indexing
        return tuple(t[idx] if isinstance(t, torch.Tensor) else t for t in obj)

def tuple_setter(obj, idx, value):
    """
    Custom setter for tuples of tensors.
    Returns a new tuple with the value set at the specified index.
    """
    result = list(obj)  # Convert to list for mutation
    
    if isinstance(idx, tuple):
        # For multi-dimensional indexing
        if len(idx) == 2 and isinstance(idx[0], slice) and idx[0] == slice(None):
            # This handles the case (slice(None), i) used in the Blelloch algorithm
            position_idx = idx[1]
            for i, (t, v) in enumerate(zip(result, value)):
                if isinstance(t, torch.Tensor) and t.dim() > 1:
                    t[:, position_idx] = v
                else:
                    if isinstance(t, torch.Tensor):
                        t[position_idx] = v
                    else:
                        # For non-tensor elements, create a new list and modify
                        t_list = list(t) if hasattr(t, '__iter__') else [t]
                        t_list[position_idx] = v
                        result[i] = type(t)(t_list) if hasattr(t, '__iter__') else v
        else:
            # General multi-dimensional case
            for i, (t, v) in enumerate(zip(result, value)):
                if isinstance(t, torch.Tensor):
                    t[idx] = v
                else:
                    # For non-tensor elements, create a new list and modify
                    t_list = list(t) if hasattr(t, '__iter__') else [t]
                    t_list[idx] = v
                    result[i] = type(t)(t_list) if hasattr(t, '__iter__') else v
    else:
        # For single-dimension indexing
        for i, (t, v) in enumerate(zip(result, value)):
            if isinstance(t, torch.Tensor):
                t[idx] = v
            else:
                # For non-tensor elements, create a new list and modify
                t_list = list(t) if hasattr(t, '__iter__') else [t]
                t_list[idx] = v
                result[i] = type(t)(t_list) if hasattr(t, '__iter__') else v
    
    return tuple(result)  # Convert back to tuple

def tuple_add(a, b):
    """Addition operator for tuples of tensors"""
    return tuple(ai + bi for ai, bi in zip(a, b))

def tuple_mul(a, b):
    """Multiplication operator for tuples of tensors"""
    return tuple(ai * bi for ai, bi in zip(a, b))

def tuple_max(a, b):
    """Element-wise maximum for tuples of tensors"""
    return tuple(torch.max(ai, bi) if isinstance(ai, torch.Tensor) else max(ai, bi) for ai, bi in zip(a, b))

def verify_scan_tuple(batch_size: int, seq_len: int, channels: int, op: Callable = tuple_add, 
                    device: str = "cpu") -> bool:
    """
    Verifies the batched Blelloch scan against the naive batched scan for tuples of tensors.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_len: Length of each sequence
        channels: Number of channels/features
        op: Binary associative operator (default: tuple_add)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create random tensors
    tensor_a = torch.rand(batch_size, seq_len, channels, device=device)
    tensor_b = torch.rand(batch_size, seq_len, channels, device=device)
    
    # Ensure values are small integers for better numerical stability in tests
    tensor_a = (tensor_a * 10).int().float()
    tensor_b = (tensor_b * 10).int().float()
    
    # Create a tuple of tensors
    tensor_tuple = (tensor_a, tensor_b)
    
    # Determine identity element
    identity_element = get_identity_element(op, tensor_tuple[0][0, 0])
    
    # Run both implementations with custom indexing and setting
    naive_result = naive_scan_batched(
        tensor_tuple, op, identity_element, dim=1,
        indexer=tuple_indexer, setter=tuple_setter
    )
    
    blelloch_result = blelloch_scan_batched(
        tensor_tuple, op, identity_element, dim=1,
        indexer=tuple_indexer, setter=tuple_setter
    )
    
    print(f"Input tensor tuple shape: ({tensor_a.shape}, {tensor_b.shape})")
    print(f"Operator: {op.__name__ if hasattr(op, '__name__') else op.__class__.__name__}")
    
    # Check shapes
    print(f"Naive result shape: ({naive_result[0].shape}, {naive_result[1].shape})")
    print(f"Blelloch result shape: ({blelloch_result[0].shape}, {blelloch_result[1].shape})")
    
    # Check if results are equal
    is_equal_a = torch.allclose(naive_result[0], blelloch_result[0], rtol=1e-5, atol=1e-5)
    is_equal_b = torch.allclose(naive_result[1], blelloch_result[1], rtol=1e-5, atol=1e-5)
    
    if is_equal_a and is_equal_b:
        print("✓ Verification PASSED: Both implementations produce identical results for tuple inputs.")
        
        # Print sample results for first batch, first channel
        print("\nSample results for first batch, first channel:")
        print(f"Input A: {tensor_a[0, :, 0].cpu().numpy()}")
        print(f"Input B: {tensor_b[0, :, 0].cpu().numpy()}")
        print(f"Naive scan A: {naive_result[0][0, :, 0].cpu().numpy()}")
        print(f"Naive scan B: {naive_result[1][0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan A: {blelloch_result[0][0, :, 0].cpu().numpy()}")
        print(f"Blelloch scan B: {blelloch_result[1][0, :, 0].cpu().numpy()}")
        
        return True
    else:
        # Find where differences occur
        diff_a = (naive_result[0] - blelloch_result[0]).abs()
        diff_b = (naive_result[1] - blelloch_result[1]).abs()
        max_diff_a = diff_a.max().item()
        max_diff_b = diff_b.max().item()
        
        print(f"✗ Verification FAILED:")
        print(f"Max difference in tensor A: {max_diff_a}")
        print(f"Max difference in tensor B: {max_diff_b}")
        
        return False

def run_tensor_tuple_tests():
    """Run tests on tuple of tensors using custom indexing and operators"""
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test with different batch sizes, sequence lengths, and channel dimensions
    print("\n==== Testing and Verification with Tuple of Tensors ====")
    
    test_configs = [
        (2, 8, 3),    # Small config: 2 batches, 8 length, 3 channels
        (4, 16, 5),   # Medium config: 4 batches, 16 length, 5 channels
    ]
    
    operators = [
        (tuple_add, "tuple_add"),
        (tuple_mul, "tuple_mul"),
        (tuple_max, "tuple_max"),
    ]
    
    for batch, length, channels in test_configs:
        print(f"\n=== Testing batch={batch}, seq_len={length}, channels={channels} ===")
        
        for op, name in operators:
            print(f"\nOperator: {name}")
            verify_scan_tuple(batch, length, channels, op, device=device)

if __name__ == "__main__":
    run_tensor_tuple_tests()