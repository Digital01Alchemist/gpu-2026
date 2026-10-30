#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <climits>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +         \
                                      cudaGetErrorString(err__));           \
        }                                                                  \
    } while (0)

namespace {

constexpr int kMaxBlockDim = 256;
constexpr int kWarpSize    = 32;

template <typename Op>
__device__ __forceinline__ float BlockReduce(float val, float* warpBuf,
                                              Op op, float identity) {
    const int lane = threadIdx.x % kWarpSize;
    const int wid  = threadIdx.x / kWarpSize;
    const int numWarps = (blockDim.x + kWarpSize - 1) / kWarpSize;

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if (lane == 0) warpBuf[wid] = val;
    __syncthreads();

    val = (threadIdx.x < numWarps) ? warpBuf[threadIdx.x] : identity;
    if (wid == 0) {
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            val = op(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) warpBuf[0] = val;
    }
    __syncthreads();
    return warpBuf[0];
}

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a > b ? a : b;
    }
};
struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a + b;
    }
};

__global__ void SoftmaxKernelShared(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int row_size) {
    extern __shared__ float srow[];           
    __shared__ float warpBuf[kMaxBlockDim / kWarpSize];

    const int row = blockIdx.x;
    const float* in  = input  + (size_t)row * row_size;
    float*       out = output + (size_t)row * row_size;

    const int tid    = threadIdx.x;
    const int stride = blockDim.x;

    float localMax = -FLT_MAX;
    for (int i = tid; i < row_size; i += stride) {
        float v = in[i];
        srow[i] = v;
        localMax = fmaxf(localMax, v);
    }
    __syncthreads();

    float rowMax = BlockReduce(localMax, warpBuf, MaxOp(), -FLT_MAX);

    float localSum = 0.f;
    for (int i = tid; i < row_size; i += stride) {
        float e = __expf(srow[i] - rowMax);
        srow[i] = e;
        localSum += e;
    }
    __syncthreads();

    float rowSum = BlockReduce(localSum, warpBuf, SumOp(), 0.f);
    float invSum = 1.0f / rowSum;

    for (int i = tid; i < row_size; i += stride) {
        out[i] = srow[i] * invSum;
    }
}

__global__ void SoftmaxKernelGlobalFallback(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int row_size) {
    __shared__ float warpBuf[kMaxBlockDim / kWarpSize];

    const int row = blockIdx.x;
    const float* in  = input  + (size_t)row * row_size;
    float*       out = output + (size_t)row * row_size;

    const int tid    = threadIdx.x;
    const int stride = blockDim.x;

    float localMax = -FLT_MAX;
    for (int i = tid; i < row_size; i += stride) {
        localMax = fmaxf(localMax, in[i]);
    }
    float rowMax = BlockReduce(localMax, warpBuf, MaxOp(), -FLT_MAX);

    float localSum = 0.f;
    for (int i = tid; i < row_size; i += stride) {
        localSum += __expf(in[i] - rowMax);
    }
    float rowSum = BlockReduce(localSum, warpBuf, SumOp(), 0.f);
    float invSum = 1.0f / rowSum;

    for (int i = tid; i < row_size; i += stride) {
        out[i] = __expf(in[i] - rowMax) * invSum;
    }
}

struct CudaResources {
    float* d_in  = nullptr;
    float* d_out = nullptr;
    size_t capacityElems = 0;
    cudaStream_t stream = nullptr;
    bool ready = false;
    int  maxOptinSharedMem = 0;

    void EnsureReady(size_t elems) {
        if (!ready) {
            CUDA_CHECK(cudaStreamCreate(&stream));

            int device;
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaDeviceGetAttribute(
                &maxOptinSharedMem,
                cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

            CUDA_CHECK(cudaFuncSetAttribute(
                SoftmaxKernelShared,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxOptinSharedMem));

            ready = true;
        }
        if (elems > capacityElems) {
            if (d_in)  CUDA_CHECK(cudaFree(d_in));
            if (d_out) CUDA_CHECK(cudaFree(d_out));
            CUDA_CHECK(cudaMalloc(&d_in,  elems * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out, elems * sizeof(float)));
            capacityElems = elems;
        }
    }

    ~CudaResources() {
        if (d_in)  cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        if (ready) cudaStreamDestroy(stream);
    }
};

CudaResources g_res;

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t total = input.size();
    std::vector<float> output(total);
    if (total == 0 || row_count == 0) return output;

    const int row_size = static_cast<int>(total / row_count);

    g_res.EnsureReady(total);
    cudaStream_t stream = g_res.stream;

    float* h_in  = const_cast<float*>(input.data());
    float* h_out = output.data();

    cudaError_t regIn  = cudaHostRegister(h_in,  total * sizeof(float), cudaHostRegisterDefault);
    cudaError_t regOut = cudaHostRegister(h_out, total * sizeof(float), cudaHostRegisterDefault);

    CUDA_CHECK(cudaMemcpyAsync(g_res.d_in, h_in, total * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    const int block = std::min(row_size, kMaxBlockDim);
    const size_t sharedBytes = (size_t)row_size * sizeof(float);

    if (sharedBytes <= (size_t)g_res.maxOptinSharedMem) {
        SoftmaxKernelShared<<<row_count, block, sharedBytes, stream>>>(
            g_res.d_in, g_res.d_out, row_size);
    } else {
        SoftmaxKernelGlobalFallback<<<row_count, block, 0, stream>>>(
            g_res.d_in, g_res.d_out, row_size);
    }

    CUDA_CHECK(cudaMemcpyAsync(h_out, g_res.d_out, total * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (regIn  == cudaSuccess) cudaHostUnregister(h_in);
    if (regOut == cudaSuccess) cudaHostUnregister(h_out);

    return output;
}