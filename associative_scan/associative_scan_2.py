import torch
import time
import math

def naive_scan(arr):
    """
    Naive sequential exclusive scan implementation using PyTorch.
    
    Args:
        arr: Input tensor
        
    Returns:
        Exclusive scan result (prefix sum where each element is the sum of all previous elements)
    """
    n = arr.size(0)
    result = torch.zeros_like(arr)
    
    for i in range(1, n):
        result[i] = result[i-1] + arr[i-1]
        
    return result

def blelloch_scan(arr):
    """
    Blelloch parallel exclusive scan implementation using PyTorch.
    This is a simulation of the parallel algorithm executed sequentially.
    
    Args:
        arr: Input tensor
        
    Returns:
        Exclusive scan result
    """
    arr = arr.clone()  # Make a copy to avoid modifying the input
    n = arr.size(0)
    
    # Round up to the next power of 2
    pow2 = 1
    while pow2 < n:
        pow2 *= 2
    
    # Pad the array if needed
    if n < pow2:
        original_n = n
        padding = torch.zeros(pow2 - n, device=arr.device, dtype=arr.dtype)
        arr = torch.cat((arr, padding))
        n = pow2
    else:
        original_n = n
    
    # Up-sweep (reduce) phase
    for d in range(int(math.log2(n))):
        step = 2 ** (d+1)
        for i in range(0, n, step):
            arr[i + step - 1] += arr[i + step//2 - 1]
    
    # Set the last element to 0 (for exclusive scan)
    arr[n-1] = 0
    
    # Down-sweep phase
    for d in range(int(math.log2(n))-1, -1, -1):
        step = 2 ** (d+1)
        for i in range(0, n, step):
            temp = arr[i + step//2 - 1].clone()
            arr[i + step//2 - 1] = arr[i + step - 1]
            arr[i + step - 1] += temp
    
    # Return only the originally sized result
    return arr[:original_n]

def native_pytorch_scan(arr):
    """
    Using PyTorch's built-in cumulative sum for comparison.
    Note: cumsum is inclusive, so we need to shift to make it exclusive.
    
    Args:
        arr: Input tensor
        
    Returns:
        Exclusive scan result
    """
    result = torch.zeros_like(arr)
    result[1:] = torch.cumsum(arr[:-1], dim=0)
    return result

def verify_scan(arr, device="cpu"):
    """
    Verifies the Blelloch scan against the naive scan and PyTorch's native scan.
    
    Args:
        arr: Input array that will be converted to a tensor
        device: "cpu" or "cuda" for GPU (if available)
        
    Returns:
        True if all scans produce identical results, False otherwise
    """
    # Convert to tensor and move to specified device
    tensor_arr = torch.tensor(arr, dtype=torch.float32, device=device)
    
    naive_result = naive_scan(tensor_arr)
    blelloch_result = blelloch_scan(tensor_arr)
    pytorch_result = native_pytorch_scan(tensor_arr)
    
    print(f"Input array: {tensor_arr.cpu().numpy()}")
    print(f"Naive scan result: {naive_result.cpu().numpy()}")
    print(f"Blelloch scan result: {blelloch_result.cpu().numpy()}")
    print(f"PyTorch native scan result: {pytorch_result.cpu().numpy()}")
    
    # Check if results are equal (with a small tolerance for floating point differences)
    naive_blelloch_equal = torch.allclose(naive_result, blelloch_result, rtol=1e-5, atol=1e-5)
    naive_pytorch_equal = torch.allclose(naive_result, pytorch_result, rtol=1e-5, atol=1e-5)
    
    if naive_blelloch_equal and naive_pytorch_equal:
        print("✓ Verification PASSED: All implementations produce identical results.")
        return True
    else:
        if not naive_blelloch_equal:
            diff = naive_result - blelloch_result
            print(f"✗ Verification FAILED: Difference between naive and Blelloch: {diff.cpu().numpy()}")
        if not naive_pytorch_equal:
            diff = naive_result - pytorch_result
            print(f"✗ Verification FAILED: Difference between naive and PyTorch: {diff.cpu().numpy()}")
        return False

def benchmark_scan(arr, iterations=100, device="cpu"):
    """
    Benchmarks all scan implementations.
    
    Args:
        arr: Input array that will be converted to a tensor
        iterations: Number of iterations for timing
        device: "cpu" or "cuda" for GPU (if available)
    """
    # Convert to tensor and move to specified device
    tensor_arr = torch.tensor(arr, dtype=torch.float32, device=device)
    
    # Warmup
    for _ in range(10):
        naive_scan(tensor_arr)
        blelloch_scan(tensor_arr)
        native_pytorch_scan(tensor_arr)
    
    # Synchronize before timing (important for GPU)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark naive scan
    start_time = time.time()
    for _ in range(iterations):
        naive_scan(tensor_arr)
    if device == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.time() - start_time) / iterations
    
    # Benchmark Blelloch scan
    start_time = time.time()
    for _ in range(iterations):
        blelloch_scan(tensor_arr)
    if device == "cuda":
        torch.cuda.synchronize()
    blelloch_time = (time.time() - start_time) / iterations
    
    # Benchmark PyTorch's native scan
    start_time = time.time()
    for _ in range(iterations):
        native_pytorch_scan(tensor_arr)
    if device == "cuda":
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / iterations
    
    print(f"\nBenchmark results on {device.upper()} (averaged over {iterations} iterations):")
    print(f"Naive scan: {naive_time * 1000:.4f} ms")
    print(f"Blelloch scan: {blelloch_time * 1000:.4f} ms")
    print(f"PyTorch native scan: {pytorch_time * 1000:.4f} ms")
    print(f"Speedup (naive vs Blelloch): {naive_time / blelloch_time:.2f}x")
    print(f"Speedup (naive vs PyTorch native): {naive_time / pytorch_time:.2f}x")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test with different arrays
test_arrays = [
    [3, 1, 7, 0, 4, 1, 6, 3],
    [1, 2, 3, 4, 5],
    [10, 20, 30, 40],
    [5],
    [],
    [torch.randint(0, 100, (15,)).numpy()]
]

print("==== Testing and Verification ====")
for i, arr in enumerate(test_arrays):
    if len(arr) > 0:  # Skip empty arrays
        print(f"\nTest {i+1}:")
        verify_scan(arr, device=device)

print("\n==== Performance Benchmark ====")
# Generate random data for benchmarking
benchmark_array = torch.randint(0, 100, (1000,)).numpy()

# Run benchmarks on CPU
benchmark_scan(benchmark_array, device="cpu")

# Run benchmarks on GPU if available
if torch.cuda.is_available():
    benchmark_scan(benchmark_array, device="cuda")
else:
    print("\nCUDA not available, skipping GPU benchmarks.")