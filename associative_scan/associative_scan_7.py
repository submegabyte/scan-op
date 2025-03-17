import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple, Optional

def naive_scan_3d(arr: torch.Tensor, op: Callable = operator.add, 
                  identity_element: Union[int, float] = 0, dim: int = 1) -> torch.Tensor:
    """
    Naive sequential exclusive scan implementation for 3D tensors [B, L, D].
    Performs scan along the specified dimension (default: 1 = sequence length).
    
    Args:
        arr: Input tensor of shape [B, L, D]
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1 = sequence length)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    result = torch.full_like(arr, identity_element)
    
    # Get shape information
    shape = arr.shape
    scan_length = shape[dim]
    
    if dim == 1:  # Most common case - scan along sequence length
        for i in range(1, scan_length):
            result[:, i, :] = op(result[:, i-1, :], arr[:, i-1, :])
    elif dim == 0:  # Scan along batch dimension
        for i in range(1, scan_length):
            result[i, :, :] = op(result[i-1, :, :], arr[i-1, :, :])
    elif dim == 2:  # Scan along channel dimension
        for i in range(1, scan_length):
            result[:, :, i] = op(result[:, :, i-1], arr[:, :, i-1])
    else:
        raise ValueError(f"Invalid dimension: {dim}. Must be 0, 1, or 2 for 3D tensor.")
        
    return result

def blelloch_scan_3d(arr: torch.Tensor, op: Callable = operator.add, 
                     identity_element: Union[int, float] = 0, dim: int = 1) -> torch.Tensor:
    """
    Blelloch parallel exclusive scan implementation for 3D tensors [B, L, D].
    Performs scan along the specified dimension (default: 1 = sequence length).
    
    Args:
        arr: Input tensor of shape [B, L, D]
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        dim: Dimension along which to perform the scan (default: 1 = sequence length)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    # Make a copy to avoid modifying the input
    result = arr.clone()
    
    # Get shape information
    shape = arr.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D tensor of shape [B, L, D], got shape {shape}")
    
    scan_length = shape[dim]
    
    # Handle empty or singleton dimension
    if scan_length <= 1:
        if scan_length == 1:
            # For exclusive scan with single element, result is identity
            if dim == 0:
                result[0, :, :] = identity_element
            elif dim == 1:
                result[:, 0, :] = identity_element
            else:  # dim == 2
                result[:, :, 0] = identity_element
        return result
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < scan_length:
        pow2 *= 2
    
    # Create an expanded tensor if needed
    padded = False
    original_length = scan_length
    
    if scan_length < pow2:
        padded = True
        pad_size = pow2 - scan_length
        
        if dim == 0:
            padding = torch.full((pad_size, shape[1], shape[2]), identity_element, 
                                 device=result.device, dtype=result.dtype)
            result = torch.cat((result, padding), dim=0)
        elif dim == 1:
            padding = torch.full((shape[0], pad_size, shape[2]), identity_element, 
                                 device=result.device, dtype=result.dtype)
            result = torch.cat((result, padding), dim=1)
        else:  # dim == 2
            padding = torch.full((shape[0], shape[1], pad_size), identity_element, 
                                 device=result.device, dtype=result.dtype)
            result = torch.cat((result, padding), dim=2)
            
        scan_length = pow2
    
    # Perform Blelloch scan along the specified dimension
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(scan_length))):
        step = 2 ** (d+1)
        
        # Create index tensors for the specified dimension
        if dim == 0:
            for i in range(0, scan_length, step):
                result[i + step - 1, :, :] = op(result[i + step - 1, :, :], 
                                               result[i + step//2 - 1, :, :])
        elif dim == 1:
            for i in range(0, scan_length, step):
                result[:, i + step - 1, :] = op(result[:, i + step - 1, :], 
                                               result[:, i + step//2 - 1, :])
        else:  # dim == 2
            for i in range(0, scan_length, step):
                result[:, :, i + step - 1] = op(result[:, :, i + step - 1], 
                                               result[:, :, i + step//2 - 1])
    
    # Set the last element to identity (for exclusive scan)
    if dim == 0:
        result[scan_length-1, :, :] = identity_element
    elif dim == 1:
        result[:, scan_length-1, :] = identity_element
    else:  # dim == 2
        result[:, :, scan_length-1] = identity_element
    
    # Down-sweep phase
    for d in range(int(math.log2(scan_length))-1, -1, -1):
        step = 2 ** (d+1)
        
        if dim == 0:
            for i in range(0, scan_length, step):
                temp = result[i + step//2 - 1, :, :].clone()
                result[i + step//2 - 1, :, :] = result[i + step - 1, :, :]
                result[i + step - 1, :, :] = op(result[i + step - 1, :, :], temp)
        elif dim == 1:
            for i in range(0, scan_length, step):
                temp = result[:, i + step//2 - 1, :].clone()
                result[:, i + step//2 - 1, :] = result[:, i + step - 1, :]
                result[:, i + step - 1, :] = op(result[:, i + step - 1, :], temp)
        else:  # dim == 2
            for i in range(0, scan_length, step):
                temp = result[:, :, i + step//2 - 1].clone()
                result[:, :, i + step//2 - 1] = result[:, :, i + step - 1]
                result[:, :, i + step - 1] = op(result[:, :, i + step - 1], temp)
    
    # Return only the original sized result if padded
    if padded:
        if dim == 0:
            return result[:original_length, :, :]
        elif dim == 1:
            return result[:, :original_length, :]
        else:  # dim == 2
            return result[:, :, :original_length]
    else:
        return result

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
    elif op is max:
        return float('-inf')
    elif op is min:
        return float('inf')
    else:
        raise ValueError("Unknown operator. Please provide an identity element.")

def batched_scan_efficient(arr: torch.Tensor, op: Callable = operator.add, 
                          identity_element: Optional[Union[int, float]] = None, 
                          dim: int = 1, algorithm: str = "blelloch") -> torch.Tensor:
    """
    Efficient implementation of scan for batched 3D tensors [B, L, D].
    This is a wrapper function that handles both naive and Blelloch scans.
    
    Args:
        arr: Input tensor of shape [B, L, D]
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 1 = sequence length)
        algorithm: "blelloch" or "naive" (default: "blelloch")
        
    Returns:
        Exclusive scan result along the specified dimension
    """
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    if algorithm.lower() == "blelloch":
        return blelloch_scan_3d(arr, op, identity_element, dim)
    elif algorithm.lower() == "naive":
        return naive_scan_3d(arr, op, identity_element, dim)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'blelloch' or 'naive'.")

def verify_scan_3d(B: int, L: int, D: int, op: Callable = operator.add, 
                   identity_element: Optional[Union[int, float]] = None, 
                   dim: int = 1, device: str = "cpu") -> bool:
    """
    Verifies the 3D Blelloch scan against the naive scan with random input.
    
    Args:
        B: Batch size
        L: Sequence length
        D: Number of channels
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 1 = sequence length)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Create random tensor
    tensor_arr = torch.rand(B, L, D, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Run both scan algorithms
    naive_result = naive_scan_3d(tensor_arr, op, identity_element, dim)
    blelloch_result = blelloch_scan_3d(tensor_arr, op, identity_element, dim)
    
    print(f"Verification for 3D tensor with shape [{B}, {L}, {D}]")
    print(f"Operator: {op.__name__}")
    print(f"Scanning along dimension: {dim}")
    
    # Check if results are equal (with a small tolerance for floating point differences)
    if torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5):
        print("✓ Verification PASSED: Both implementations produce identical results.")
        return True
    else:
        diff = torch.abs(naive_result - blelloch_result).max().item()
        print(f"✗ Verification FAILED: Maximum absolute difference: {diff}")
        return False

def benchmark_scan_3d(B: int, L: int, D: int, op: Callable = operator.add, 
                     identity_element: Optional[Union[int, float]] = None, 
                     dim: int = 1, iterations: int = 20, device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the 3D scan implementations.
    
    Args:
        B: Batch size
        L: Sequence length
        D: Number of channels
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        dim: Dimension along which to perform the scan (default: 1 = sequence length)
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Create random tensor
    tensor_arr = torch.rand(B, L, D, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Warmup
    for _ in range(5):
        naive_scan_3d(tensor_arr, op, identity_element, dim)
        blelloch_scan_3d(tensor_arr, op, identity_element, dim)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan_3d(tensor_arr, op, identity_element, dim)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan_3d(tensor_arr, op, identity_element, dim)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results for shape [{B}, {L}, {D}] on {device.upper()}")
    print(f"Operator: {op.__name__}, Dimension: {dim}")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different shapes and dimensions
print("\n==== Testing and Verification of 3D Batched Scans ====")

shapes = [
    (2, 8, 4),    # Small shape
    (8, 16, 32),  # Medium shape
    (16, 32, 64)  # Larger shape
]

operators = [
    (operator.add, "Addition"),
    (operator.mul, "Multiplication"),
    (max, "Maximum")
]

print("\n--- Verification Tests ---")
for B, L, D in shapes:
    print(f"\nShape: [{B}, {L}, {D}]")
    for op, op_name in operators:
        for dim in range(3):
            verify_scan_3d(B, L, D, op, None, dim, device)

print("\n--- Performance Benchmarks ---")
for B, L, D in shapes:
    print(f"\nShape: [{B}, {L}, {D}]")
    for op, op_name in operators:
        for dim in range(3):
            benchmark_scan_3d(B, L, D, op, None, dim, 10, device)

# Example usage with real data
def example_usage():
    """
    Example demonstrating how to use the batched scan on some sample data.
    """
    print("\n==== Example Usage ====")
    
    # Create sample data - batch of 4 sequences, each with length 10 and 3 channels
    batch_size, seq_len, channels = 4, 10, 3
    data = torch.rand(batch_size, seq_len, channels, device=device)
    
    print(f"Input shape: {data.shape}")
    print("First sequence in batch:\n", data[0])
    
    # Perform cumulative sum (exclusive) along sequence dimension
    sum_result = batched_scan_efficient(data, operator.add, 0, dim=1)
    print("\nCumulative sum along sequence dimension (exclusive):")
    print("Result shape:", sum_result.shape)
    print("First sequence result:\n", sum_result[0])
    
    # Perform running maximum along channel dimension
    max_result = batched_scan_efficient(data, max, float('-inf'), dim=2)
    print("\nRunning maximum along channel dimension (exclusive):")
    print("Result shape:", max_result.shape)
    print("First sequence result:\n", max_result[0])
    
    # Perform running product along batch dimension
    prod_result = batched_scan_efficient(data, operator.mul, 1, dim=0)
    print("\nRunning product along batch dimension (exclusive):")
    print("Result shape:", prod_result.shape)
    print("First sequence result:\n", prod_result[0])

# Run the example
example_usage()