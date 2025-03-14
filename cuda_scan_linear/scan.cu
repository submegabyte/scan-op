#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024  // Adjust as needed

// Custom Associative Scan Kernel
__global__ void associative_scan_kernel(
    const float *sx, const float *sy,  // Inputs (sx, sy)
    const float *cx, const float *cy,  // Inputs (cx, cy)
    float *Sx_out, float *Sy_out,  // Outputs (Sx, Sy)
    int n
) {
    __shared__ float temp_sx[BLOCK_SIZE];  
    __shared__ float temp_sy[BLOCK_SIZE];  
    __shared__ float temp_cx[BLOCK_SIZE];  
    __shared__ float temp_cy[BLOCK_SIZE];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        temp_sx[threadIdx.x] = sx[tid];
        temp_sy[threadIdx.x] = sy[tid];
        temp_cx[threadIdx.x] = cx[tid];
        temp_cy[threadIdx.x] = cy[tid];
    }
    __syncthreads();

    // Perform the associative scan (in-place)
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (threadIdx.x >= offset) {
            float prev_sx = temp_sx[threadIdx.x - offset];
            float prev_sy = temp_sy[threadIdx.x - offset];

            temp_sx[threadIdx.x] = temp_cx[threadIdx.x] * prev_sx;
            temp_sy[threadIdx.x] = temp_cx[threadIdx.x] * prev_sy + temp_cy[threadIdx.x];
        }
        __syncthreads();
    }

    // Write results to global memory
    if (tid < n) {
        Sx_out[tid] = temp_sx[threadIdx.x];
        Sy_out[tid] = temp_sy[threadIdx.x];
    }
}

// Python binding for PyTorch
std::vector<torch::Tensor> associative_scan(torch::Tensor sx, torch::Tensor sy, torch::Tensor cx, torch::Tensor cy) {
    auto Sx_out = torch::empty_like(sx);
    auto Sy_out = torch::empty_like(sy);

    const int n = sx.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;

    associative_scan_kernel<<<blocks, threads>>>(
        sx.data_ptr<float>(), sy.data_ptr<float>(),
        cx.data_ptr<float>(), cy.data_ptr<float>(),
        Sx_out.data_ptr<float>(), Sy_out.data_ptr<float>(),
        n
    );

    return {Sx_out, Sy_out};
}

// Bind function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("associative_scan", &associative_scan, "Associative Scan (CUDA)");
}
