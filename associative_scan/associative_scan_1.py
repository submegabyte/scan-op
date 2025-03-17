import numpy as np
import time

def naive_scan(arr):
    """
    Naive sequential exclusive scan implementation.
    
    Args:
        arr: Input array
        
    Returns:
        Exclusive scan result (prefix sum where each element is the sum of all previous elements)
    """
    n = len(arr)
    result = np.zeros_like(arr)
    
    for i in range(1, n):
        result[i] = result[i-1] + arr[i-1]
        
    return result

def blelloch_scan(arr):
    """
    Blelloch parallel exclusive scan implementation.
    This is a simulation of the parallel algorithm executed sequentially.
    
    Args:
        arr: Input array
        
    Returns:
        Exclusive scan result
    """
    arr = np.copy(arr)  # Make a copy to avoid modifying the input
    n = len(arr)
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < n:
        pow2 *= 2
    
    # Pad the array if needed
    if n < pow2:
        original_n = n
        padding = np.zeros(pow2 - n)
        arr = np.concatenate((arr, padding))
        n = pow2
    else:
        original_n = n
    
    # Up-sweep (reduce) phase
    for d in range(int(np.log2(n))):
        step = 2 ** (d+1)
        for i in range(0, n, step):
            arr[i + step - 1] += arr[i + step//2 - 1]
    
    # Set the last element to 0 (for exclusive scan)
    arr[n-1] = 0
    
    # Down-sweep phase
    for d in range(int(np.log2(n))-1, -1, -1):
        step = 2 ** (d+1)
        for i in range(0, n, step):
            temp = arr[i + step//2 - 1]
            arr[i + step//2 - 1] = arr[i + step - 1]
            arr[i + step - 1] += temp
    
    # Return only the originally sized result
    return arr[:original_n]

def verify_scan(arr):
    """
    Verifies the Blelloch scan against the naive scan.
    
    Args:
        arr: Input array
        
    Returns:
        True if both scans produce identical results, False otherwise
    """
    naive_result = naive_scan(arr)
    blelloch_result = blelloch_scan(arr)
    
    print(f"Input array: {arr}")
    print(f"Naive scan result: {naive_result}")
    print(f"Blelloch scan result: {blelloch_result}")
    
    # Check if results are equal
    if np.array_equal(naive_result, blelloch_result):
        print("✓ Verification PASSED: Both implementations produce identical results.")
        return True
    else:
        diff = naive_result - blelloch_result
        print(f"✗ Verification FAILED: Difference between results: {diff}")
        return False

def benchmark_scan(arr, iterations=100):
    """
    Benchmarks both scan implementations.
    
    Args:
        arr: Input array
        iterations: Number of iterations for timing
    """
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan(arr)
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan(arr)
    blelloch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results (averaged over {iterations} iterations):")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"Speedup factor: {naive_time / blelloch_time:.2f}x")

# Test with different arrays
test_arrays = [
    np.array([3, 1, 7, 0, 4, 1, 6, 3]),
    np.array([1, 2, 3, 4, 5]),
    np.array([10, 20, 30, 40]),
    np.array([5]),
    np.array([]),
    np.random.randint(0, 100, size=15)
]

print("==== Testing and Verification ====")
for i, arr in enumerate(test_arrays):
    print(f"\nTest {i+1}:")
    verify_scan(arr)

print("\n==== Performance Benchmark ====")
benchmark_array = np.random.randint(0, 100, size=1000)
benchmark_scan(benchmark_array)