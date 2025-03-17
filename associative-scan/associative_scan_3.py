import torch
import time
import math
import operator
from typing import Callable, Union, List, Tuple

def naive_scan(arr: torch.Tensor, op: Callable = operator.add, identity_element: Union[int, float] = 0) -> torch.Tensor:
    """
    Naive sequential exclusive scan implementation using PyTorch.
    
    Args:
        arr: Input tensor
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    n = arr.size(0)
    result = torch.full_like(arr, identity_element)
    
    for i in range(1, n):
        result[i] = op(result[i-1], arr[i-1])
        
    return result

def blelloch_scan(arr: torch.Tensor, op: Callable = operator.add, identity_element: Union[int, float] = 0) -> torch.Tensor:
    """
    Blelloch parallel exclusive scan implementation using PyTorch.
    Works with any associative operator.
    
    Args:
        arr: Input tensor
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (default: 0 for addition)
        
    Returns:
        Exclusive scan result using the specified operator
    """
    arr = arr.clone()  # Make a copy to avoid modifying the input
    n = arr.size(0)
    
    # Handle empty input
    if n == 0:
        return arr
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < n:
        pow2 *= 2
    
    # Pad the array if needed
    if n < pow2:
        original_n = n
        padding = torch.full((pow2 - n,), identity_element, device=arr.device, dtype=arr.dtype)
        arr = torch.cat((arr, padding))
        n = pow2
    else:
        original_n = n
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(n))):
        step = 2 ** (d+1)
        for i in range(0, n, step):
            arr[i + step - 1] = op(arr[i + step - 1], arr[i + step//2 - 1])
    
    # Set the last element to identity element (for exclusive scan)
    arr[n-1] = identity_element
    
    # Down-sweep phase
    for d in range(int(math.log2(n))-1, -1, -1):
        step = 2 ** (d+1)
        for i in range(0, n, step):
            temp = arr[i + step//2 - 1].clone()
            arr[i + step//2 - 1] = arr[i + step - 1]
            arr[i + step - 1] = op(arr[i + step - 1], temp)
    
    # Return only the originally sized result
    return arr[:original_n]

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

def verify_scan(arr: List[Union[int, float]], op: Callable = operator.add, 
                identity_element: Union[int, float] = None, device: str = "cpu") -> bool:
    """
    Verifies the generalized Blelloch scan against the naive scan.
    
    Args:
        arr: Input array that will be converted to a tensor
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    # Convert to tensor and move to specified device
    tensor_arr = torch.tensor(arr, dtype=torch.float32, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    naive_result = naive_scan(tensor_arr, op, identity_element)
    blelloch_result = blelloch_scan(tensor_arr, op, identity_element)
    
    print(f"Input array: {tensor_arr.cpu().numpy()}")
    print(f"Operator: {op.__name__}")
    print(f"Identity element: {identity_element}")
    print(f"Naive scan result: {naive_result.cpu().numpy()}")
    print(f"Blelloch scan result: {blelloch_result.cpu().numpy()}")
    
    # Check if results are equal (with a small tolerance for floating point differences)
    if torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5):
        print("✓ Verification PASSED: Both implementations produce identical results.")
        return True
    else:
        diff = naive_result - blelloch_result
        print(f"✗ Verification FAILED: Difference between implementations: {diff.cpu().numpy()}")
        return False

def benchmark_scan(arr: List[Union[int, float]], op: Callable = operator.add, 
                  identity_element: Union[int, float] = None, iterations: int = 100, 
                  device: str = "cpu") -> Tuple[float, float]:
    """
    Benchmarks the generalized scan implementations.
    
    Args:
        arr: Input array that will be converted to a tensor
        op: Binary associative operator (default: addition)
        identity_element: Identity element for the operator (defaults to None and will be determined automatically)
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        Tuple of (naive_time, blelloch_time) in milliseconds
    """
    # Convert to tensor and move to specified device
    tensor_arr = torch.tensor(arr, dtype=torch.float32, device=device)
    
    # Determine identity element if not provided
    if identity_element is None:
        try:
            identity_element = get_identity_element(op)
        except ValueError:
            raise ValueError("Please provide an identity element for the custom operator.")
    
    # Warmup
    for _ in range(10):
        naive_scan(tensor_arr, op, identity_element)
        blelloch_scan(tensor_arr, op, identity_element)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan(tensor_arr, op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan(tensor_arr, op, identity_element)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results on {device.upper()} with operator {op.__name__} (averaged over {iterations} iterations):")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup: {naive_time / blelloch_time:.2f}x")
    
    return (naive_time * 1000, blelloch_time * 1000)

# Custom operators examples
def gcd(a, b):
    """Greatest common divisor."""
    if b == 0:
        return a
    return gcd(b, a % b)

def string_concat(a, b):
    """String concatenation operator for tensors of strings."""
    return a + b

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different operators and arrays
print("==== Testing and Verification with Different Operators ====")

# Test with addition (sum scan)
print("\n=== Addition (Sum) ===")
verify_scan([3, 1, 7, 0, 4, 1, 6, 3], operator.add, 0, device)

# Test with multiplication (product scan)
print("\n=== Multiplication (Product) ===")
verify_scan([2, 3, 1, 4, 2, 1, 3], operator.mul, 1, device)

# Test with maximum
print("\n=== Maximum ===")
verify_scan([5, 2, 8, 1, 9, 3, 7], max, float('-inf'), device)

# Test with minimum
print("\n=== Minimum ===")
verify_scan([5, 2, 8, 1, 9, 3, 7], min, float('inf'), device)

# Test with bitwise OR (for integers)
print("\n=== Bitwise OR ===")
int_arr = [1, 2, 4, 8, 16, 32, 64]
tensor_arr = torch.tensor(int_arr, dtype=torch.int32, device=device)
naive_result = naive_scan(tensor_arr, operator.or_, 0)
blelloch_result = blelloch_scan(tensor_arr, operator.or_, 0)
print(f"Input array: {tensor_arr.cpu().numpy()}")
print(f"Naive scan result: {naive_result.cpu().numpy()}")
print(f"Blelloch scan result: {blelloch_result.cpu().numpy()}")

# Benchmark with different array sizes
print("\n==== Performance Benchmarks with Different Sizes ====")
sizes = [100, 1000, 10000]
operators = [
    (operator.add, "Addition", 0),
    (operator.mul, "Multiplication", 1),
    (max, "Maximum", float('-inf')),
    (min, "Minimum", float('inf'))
]

for size in sizes:
    print(f"\n--- Array Size: {size} ---")
    benchmark_array = torch.randint(1, 10, (size,)).numpy().tolist()
    
    for op, name, identity in operators:
        print(f"\nOperator: {name}")
        benchmark_scan(benchmark_array, op, identity, iterations=20, device=device)