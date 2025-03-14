#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024  // Adjust as needed

// Custom Associative Scan Kernel using Blelloch
__global__ void blelloch_associative_scan_kernel(
    float *sx, float *sy,  // Inputs (sx, sy)
    float *cx, float *cy,  // Inputs (cx, cy)
    int n
) {
    __shared__ float temp_sx[BLOCK_SIZE];  
    __shared__ float temp_sy[BLOCK_SIZE];  
    __shared__ float temp_cx[BLOCK_SIZE];  
    __shared__ float temp_cy[BLOCK_SIZE];  

    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    // Load elements into shared memory
    if (gid < n) {
        temp_sx[tid] = sx[gid];
        temp_sy[tid] = sy[gid];
        temp_cx[tid] = cx[gid];
        temp_cy[tid] = cy[gid];
    }
    __syncthreads();

    // **UPSweep Phase (Reduction)**
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            temp_sx[idx] = temp_cx[idx] * temp_sx[idx - offset];
            temp_sy[idx] = temp_cx[idx] * temp_sy[idx - offset] + temp_cy[idx];
        }
        __syncthreads();
    }

    // Clear the last element for exclusive scan
    if (tid == 0) {
        temp_sx[blockDim.x - 1] = 1;  
        temp_sy[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // **Downsweep Phase**
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            float prev_sx = temp_sx[idx - offset];
            float prev_sy = temp_sy[idx - offset];

            temp_sx[idx - offset] = temp_cx[idx - offset] * prev_sx;
            temp_sy[idx - offset] = temp_cx[idx - offset] * prev_sy + temp_cy[idx - offset];

            temp_sx[idx] = temp_cx[idx] * temp_sx[idx];
            temp_sy[idx] = temp_cx[idx] * temp_sy[idx] + temp_cy[idx];
        }
        __syncthreads();
    }

    // Store results back to global memory
    if (gid < n) {
        sx[gid] = temp_sx[tid];
        sy[gid] = temp_sy[tid];
    }
}

// Wrapper function for PyTorch
std::vector<torch::Tensor> blelloch_associative_scan(torch::Tensor sx, torch::Tensor sy, torch::Tensor cx, torch::Tensor cy) {
    const int n = sx.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;

    blelloch_associative_scan_kernel<<<blocks, threads>>>(
        sx.data_ptr<float>(), sy.data_ptr<float>(),
        cx.data_ptr<float>(), cy.data_ptr<float>(),
        n
    );

    return {sx, sy};
}

// Bind function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blelloch_associative_scan", &blelloch_associative_scan, "Blelloch Associative Scan (CUDA)");
}
