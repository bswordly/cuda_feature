#include "sobel.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

#define TILE_DIM 16
#define BLOCK_DIM (TILE_DIM + 2)

#define CUDA_CHECK(call)                                        \
	   do {                                                       \
			 cudaError_t err = call;                                  \
			 if (err != cudaSuccess) {                                \
				 throw std::runtime_error(std::string("CUDA Error: ") + \
						 cudaGetErrorString(err));                          \
			 }                                                        \
			} while (0)                                               \

__global__ void sobel_tiled_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int width, int height) {
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output coordinates for this thread
    int col = blockIdx.x * TILE_DIM + tx;
    int row = blockIdx.y * TILE_DIM + ty;

    // Input coordinates shifted by -1 for the halo border
    int in_col = col - 1;
    int in_row = row - 1;

    // Cooperatively load tile + halo into shared memory with clamping at boundaries
    if (tx < BLOCK_DIM && ty < BLOCK_DIM) {
        int clamped_c = min(max(in_col, 0), width - 1);
        int clamped_r = min(max(in_row, 0), height - 1);
        tile[ty][tx] = in[clamped_r * width + clamped_c];
    }

    __syncthreads();

    // Only interior TILE_DIM x TILE_DIM threads compute output pixels
    if (tx < TILE_DIM && ty < TILE_DIM && col < width && row < height) {
        int sc = tx + 1;
        int sr = ty + 1;

        float gx = -tile[sr - 1][sc - 1] + tile[sr - 1][sc + 1]
                   - 2.0f * tile[sr][sc - 1] + 2.0f * tile[sr][sc + 1]
                   - tile[sr + 1][sc - 1] + tile[sr + 1][sc + 1];

        float gy = -tile[sr - 1][sc - 1] - 2.0f * tile[sr - 1][sc] - tile[sr - 1][sc + 1]
                   + tile[sr + 1][sc - 1] + 2.0f * tile[sr + 1][sc] + tile[sr + 1][sc + 1];

        out[row * width + col] = sqrtf(gx * gx + gy * gy);
    }
}

void compute_sobel(const float* input, float* output, int width, int height) {
    size_t bytes = static_cast<size_t>(width) * height * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
              (height + TILE_DIM - 1) / TILE_DIM);

    sobel_tiled_kernel<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

bool is_cuda_enabled() {
    return true;
}

