#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024  // Adjust as needed

__global__ void inclusive_scan_kernel(const int *d_in, int *d_out, int n) {
    __shared__ int temp[BLOCK_SIZE];  // Shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n)
        temp[threadIdx.x] = d_in[tid];
    __syncthreads();

    // Up-sweep (Reduction phase)
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (threadIdx.x >= offset)
            temp[threadIdx.x] += temp[threadIdx.x - offset];
        __syncthreads();
    }

    if (tid < n)
        d_out[tid] = temp[threadIdx.x];
}

torch::Tensor inclusive_scan(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int n = input.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;

    inclusive_scan_kernel<<<blocks, threads>>>(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inclusive_scan", &inclusive_scan, "Inclusive Scan (CUDA)");
}
