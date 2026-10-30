#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +         \
                                      cudaGetErrorString(err__));           \
        }                                                                  \                                                           
    } while (0)

namespace {

constexpr int kTile       = 32;  
constexpr int kThreadTile = 4;   
constexpr int kBlockDimX  = kTile;            
constexpr int kBlockDimY  = kTile / kThreadTile; 

__global__ void BlockGemmKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int n) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int tx = threadIdx.x; 
    const int ty = threadIdx.y; 

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    
    const int loadRow  = tid / 8;               
    const int loadCol4 = (tid % 8) * 4;         

    float acc[kThreadTile] = {0.f, 0.f, 0.f, 0.f};

    const int numTiles = n / kTile;

    for (int t = 0; t < numTiles; ++t) {
        const float4* aSrc = reinterpret_cast<const float4*>(
            A + (size_t)(blockRow * kTile + loadRow) * n + t * kTile + loadCol4);
        *reinterpret_cast<float4*>(&As[loadRow][loadCol4]) = *aSrc;

        const float4* bSrc = reinterpret_cast<const float4*>(
            B + (size_t)(t * kTile + loadRow) * n + blockCol * kTile + loadCol4);
        *reinterpret_cast<float4*>(&Bs[loadRow][loadCol4]) = *bSrc;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            float bElem = Bs[kk][tx];  
            #pragma unroll
            for (int i = 0; i < kThreadTile; ++i) {
                acc[i] += As[ty * kThreadTile + i][kk] * bElem;
            }
        }

        __syncthreads();
    }

    const int cRowBase = blockRow * kTile + ty * kThreadTile;
    const int cCol      = blockCol * kTile + tx;

    #pragma unroll
    for (int i = 0; i < kThreadTile; ++i) {
        C[(size_t)(cRowBase + i) * n + cCol] = acc[i];
    }
}

__global__ void NaiveGemmKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float acc = 0.f;
    for (int k = 0; k < n; ++k) {
        acc += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = acc;
}

struct CudaResources {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t capacityElems = 0;
    cudaStream_t stream = nullptr;
    bool ready = false;

    void EnsureCapacity(size_t elems) {
        if (!ready) {
            CUDA_CHECK(cudaStreamCreate(&stream));
            ready = true;
        }
        if (elems > capacityElems) {
            if (d_a) CUDA_CHECK(cudaFree(d_a));
            if (d_b) CUDA_CHECK(cudaFree(d_b));
            if (d_c) CUDA_CHECK(cudaFree(d_c));
            CUDA_CHECK(cudaMalloc(&d_a, elems * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_b, elems * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_c, elems * sizeof(float)));
            capacityElems = elems;
        }
    }

    ~CudaResources() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        if (ready) cudaStreamDestroy(stream);
    }
};

CudaResources g_res;

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    const size_t elems = (size_t)n * (size_t)n;
    std::vector<float> c(elems);
    if (n == 0) return c;

    g_res.EnsureCapacity(elems);
    cudaStream_t stream = g_res.stream;

    float* h_a = const_cast<float*>(a.data());
    float* h_b = const_cast<float*>(b.data());
    float* h_c = c.data();

    cudaError_t regA = cudaHostRegister(h_a, elems * sizeof(float), cudaHostRegisterDefault);
    cudaError_t regB = cudaHostRegister(h_b, elems * sizeof(float), cudaHostRegisterDefault);
    cudaError_t regC = cudaHostRegister(h_c, elems * sizeof(float), cudaHostRegisterDefault);

    CUDA_CHECK(cudaMemcpyAsync(g_res.d_a, h_a, elems * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(g_res.d_b, h_b, elems * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    if (n >= kTile && n % kTile == 0) {
        dim3 block(kBlockDimX, kBlockDimY);          
        dim3 grid(n / kTile, n / kTile);
        BlockGemmKernel<<<grid, block, 0, stream>>>(g_res.d_a, g_res.d_b, g_res.d_c, n);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        NaiveGemmKernel<<<grid, block, 0, stream>>>(g_res.d_a, g_res.d_b, g_res.d_c, n);
    }

    CUDA_CHECK(cudaMemcpyAsync(h_c, g_res.d_c, elems * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (regA == cudaSuccess) cudaHostUnregister(h_a);
    if (regB == cudaSuccess) cudaHostUnregister(h_b);
    if (regC == cudaSuccess) cudaHostUnregister(h_c);

    return c;
}