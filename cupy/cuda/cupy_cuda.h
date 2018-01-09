// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUDA_H
#define INCLUDE_GUARD_CUPY_CUDA_H

#include "cupy_stdint.h"

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)
#include <cuda.h>

#ifdef __APPLE__
#if CUDA_VERSION == 7050
// To avoid redefinition error of cudaDataType_t
// caused by including library_types.h.
// https://github.com/pfnet/chainer/issues/1700
// https://github.com/pfnet/chainer/pull/1819
#define __LIBRARY_TYPES_H__
#endif // #if CUDA_VERSION == 7050
#endif // #ifdef __APPLE__

#endif  // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)


#if CUPY_USE_HIP

#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <hiprand/hiprand.h>

#define CUDA_VERSION 0

extern "C" {

bool hip_environment = true;

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int CUdevice;
typedef hipError_t CUresult;
const CUresult CUDA_SUCCESS=static_cast<CUresult>(0);
enum CUjit_option {};
enum CUjitInputType {};


typedef hipDeviceptr_t CUdeviceptr;
//struct CUevent_st;
//struct CUfunc_st;
//struct CUmod_st;
struct CUlinkState_st;


typedef hipCtx_t CUcontext;
typedef hipEvent_t cudaEvent_t;
typedef hipFunction_t CUfunction;
typedef hipModule_t CUmodule;
typedef hipStream_t cudaStream_t;
typedef struct CUlinkState_st* CUlinkState;


// Error handling
CUresult cuGetErrorName(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorName(hipError);
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult hipError, const char** pStr) {
    *pStr = hipGetErrorString(hipError);
    return CUDA_SUCCESS;
}

// Primary context management
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    return hipDevicePrimaryCtxRelease(dev);
}

// Context management
CUresult cuCtxGetCurrent(CUcontext *ctx) {
    return hipCtxGetCurrent(ctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    return hipCtxSetCurrent(ctx);
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    return hipCtxCreate(pctx, flags, dev);
}

CUresult cuCtxDestroy(CUcontext ctx) {
    return hipCtxDestroy(ctx);
}



// Module load and kernel execution
CUresult cuLinkCreate(...) {
    return hipErrorUnknown;
}

CUresult cuLinkAddData(...) {
    return hipErrorUnknown;
}

CUresult cuLinkComplete(...) {
    return hipErrorUnknown;
}

CUresult cuLinkDestroy(...) {
    return hipErrorUnknown;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    return hipModuleLoad(module, fname);
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    return hipModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module) {
    return hipModuleUnload(module);
}

CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                             const char *kname) {
    return hipModuleGetFunction(function, module, kname);
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                           const char *name) {
    return hipModuleGetGlobal(dptr, bytes, hmod, name);
}

CUresult cuLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
                        uint32_t gridDimZ, uint32_t blockDimX,
                        uint32_t blockDimY, uint32_t blockDimZ,
                        uint32_t sharedMemBytes, cudaStream_t hStream,
                        void **kernelParams, void **extra) {
    return hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                 blockDimX, blockDimY, blockDimZ,
                                 sharedMemBytes, hStream, kernelParams, extra);
}


///////////////////////////////////////////////////////////////////////////////
// cuda_runtime.h
///////////////////////////////////////////////////////////////////////////////

enum {
    cudaDevAttrComputeCapabilityMajor
        = hipDeviceAttributeComputeCapabilityMajor,
    cudaDevAttrComputeCapabilityMinor
        = hipDeviceAttributeComputeCapabilityMinor,
};

typedef hipError_t cudaError_t;
const CUresult cudaSuccess=static_cast<CUresult>(0);
typedef enum {} cudaDataType;
typedef hipDeviceAttribute_t cudaDeviceAttr;
enum cudaMemoryAdvise {};
typedef hipMemcpyKind cudaMemcpyKind;


typedef hipStreamCallback_t cudaStreamCallback_t;
typedef hipPointerAttribute_t cudaPointerAttributes;


// Error handling
const char* cudaGetErrorName(cudaError_t hipError) {
    return hipGetErrorName(hipError);
}

const char* cudaGetErrorString(cudaError_t hipError) {
    return hipGetErrorString(hipError);
}


// Initialization
cudaError_t cudaDriverGetVersion(int *driverVersion) {
    return hipDriverGetVersion(driverVersion);
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    return hipRuntimeGetVersion(runtimeVersion);
}


// CUdevice operations
cudaError_t cudaGetDevice(int *deviceId) {
    return hipGetDevice(deviceId);
}

cudaError_t cudaDeviceGetAttribute(int* pi, cudaDeviceAttr attr,
                                   int deviceId) {
    return hipDeviceGetAttribute(pi, attr, deviceId);
}

cudaError_t cudaGetDeviceCount(int *count) {
    return hipGetDeviceCount(count);
}

cudaError_t cudaSetDevice(int deviceId) {
    return hipSetDevice(deviceId);
}

cudaError_t cudaDeviceSynchronize() {
    return hipDeviceSynchronize();
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int deviceId,
                                    int peerDeviceId) {
    return hipDeviceCanAccessPeer(canAccessPeer, deviceId, peerDeviceId);
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
    return hipDeviceEnablePeerAccess(peerDeviceId, flags);
}


// Memory management
cudaError_t cudaMalloc(void** ptr, size_t size) {
    return hipMalloc(ptr, size);
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipHostMalloc(ptr, size, flags);
}

cudaError_t cudaMallocManaged(...) {
    return hipErrorUnknown;
}

int cudaFree(void* ptr) {
    return hipFree(ptr);
}

cudaError_t cudaFreeHost(void* ptr) {
    return hipHostFree(ptr);
}

int cudaMemGetInfo(size_t* free, size_t* total) {
    return hipMemGetInfo(free, total);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t sizeBytes,
                       hipMemcpyKind kind) {
    return hipMemcpy(dst, src, sizeBytes, kind);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    return hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

cudaError_t cudaMemcpyPeer(void* dst, int dstDeviceId, const void* src,
                           int srcDeviceId, size_t sizeBytes) {
    return hipMemcpyPeer(dst, dstDeviceId, src, srcDeviceId, sizeBytes);
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                int srcDevice, size_t sizeBytes,
                                cudaStream_t stream) {
    return hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, sizeBytes,
                              stream);
}

cudaError_t cudaMemset(void* dst, int value, size_t sizeBytes) {
    return hipMemset(dst, value, sizeBytes);
}

cudaError_t cudaMemsetAsync(void* dst, int value, size_t sizeBytes,
                            cudaStream_t stream) {
    return hipMemsetAsync(dst, value, sizeBytes, stream);
}

cudaError_t cudaMemAdvise(...) {
    return hipErrorUnknown;
}

cudaError_t cudaMemPrefetchAsync(...) {
    return hipErrorUnknown;
}

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes,
                                     const void* ptr) {
    return hipPointerGetAttributes(attributes, ptr);
}


// Stream and Event
cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    return hipStreamCreate(stream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
                                      unsigned int flags) {
    return hipStreamCreateWithFlags(stream, flags);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return hipStreamDestroy(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return hipStreamSynchronize(stream);
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags) {
    return hipStreamAddCallback(stream, callback, userData, flags);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return hipStreamQuery(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags) {
    return hipStreamWaitEvent(stream, event, flags);
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return hipEventCreate(event);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    return hipEventCreateWithFlags(event, flags);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return hipEventDestroy(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t stop){
    return hipEventElapsedTime(ms, start, stop);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    return hipEventQuery(event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return hipEventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return hipEventSynchronize(event);
}

///////////////////////////////////////////////////////////////////////////////
// cuComplex.h
///////////////////////////////////////////////////////////////////////////////

#include "cupy_cuComplex.h"

///////////////////////////////////////////////////////////////////////////////
// blas
///////////////////////////////////////////////////////////////////////////////

typedef hipblasHandle_t cublasHandle_t;

typedef hipblasDiagType_t cublasDiagType_t;
typedef hipblasFillMode_t cublasFillMode_t;
typedef hipblasOperation_t cublasOperation_t;
typedef hipblasPointerMode_t cublasPointerMode_t;
typedef hipblasSideMode_t cublasSideMode_t;
typedef enum {} cublasGemmAlgo_t;
typedef enum {} cublasMath_t;
typedef hipblasStatus_t cublasStatus_t;

static hipblasOperation_t convert_hipblasOperation_t(hipblasOperation_t op) {
    return static_cast<hipblasOperation_t>(static_cast<int>(op) + 111);
}

// Context
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    return hipblasCreate(handle);
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return hipblasDestroy(handle);
}

cublasStatus_t cublasGetVersion(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    return hipblasSetPointerMode(handle, mode);
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode) {
    return hipblasGetPointerMode(handle, mode);
}

// Stream
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    return hipblasSetStream(handle, streamId);
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId) {
    return hipblasGetStream(handle, streamId);
}

// Math Mode
cublasStatus_t cublasSetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    return hipblasIsamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIsamin(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSasum(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSaxpy(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDaxpy(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSdot(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDdot(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCdotu(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCdotc(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZdotc(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZdotu(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSnrm2(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSscal(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


// BLAS Level 2
cublasStatus_t cublasSgemv(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgemv(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


cublasStatus_t cublasCgemv(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemv(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSger(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDger(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

int cublasCgeru(...) {
    return 0;
}

int cublasCgerc(...) {
    return 0;
}

int cublasZgeru(...) {
    return 0;
}

int cublasZgerc(...) {
    return 0;
}

// BLAS Level 3
cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k, const float *alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            const float *beta,
                            float *C, int ldc) {
    return hipblasSgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k, const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta, double *C, int ldc) {
    return hipblasDgemm(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


cublasStatus_t cublasCgemm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const float *alpha,
        const float *A[], int lda,
        const float *B[], int ldb,
        const float *beta,
        float *C[], int ldc, int batchCount) {
    return hipblasSgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

cublasStatus_t cublasDgemmBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,  const double *alpha,
        const double *A[], int lda,
        const double *B[], int ldb,
        const double *beta,
        double *C[], int ldc, int batchCount) {
    return hipblasDgemmBatched(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

cublasStatus_t cublasCgemmBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemmBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgemmEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasStrsm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDtrsm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}


// BLAS extension
cublasStatus_t cublasSgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const float *alpha,
        const float *A, int lda, const float *beta, const float *B, int ldb,
        float *C, int ldc) {
    return hipblasSgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasDgeam(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, const double *alpha,
        const double *A, int lda, const double *beta, const double *B, int ldb,
        double *C, int ldc) {
    return hipblasDgeam(handle, convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb), m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasSdgmm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgetrfBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasSgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float *alpha,
        const float *A, int lda, long long bsa,
        const float *B, int ldb, long long bsb, const float *beta,
        float *C, int ldc, long long bsc, int batchCount) {
    return hipblasSgemmStridedBatched(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k, alpha,  A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc,
        batchCount);
}

cublasStatus_t cublasDgemmStridedBatched(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const double *alpha,
        const double *A, int lda, long long bsa,
        const double *B, int ldb, long long bsb, const double *beta,
        double *C, int ldc, long long bsc, int batchCount) {
    return hipblasDgemmStridedBatched(
        handle,
        convert_hipblasOperation_t(transa), convert_hipblasOperation_t(transb),
        m, n, k, alpha,  A, lda, bsa, B, ldb, bsb, beta, C, ldc, bsc,
        batchCount);
}

cublasStatus_t cublasCgemmStridedBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemmStridedBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

///////////////////////////////////////////////////////////////////////////////
// rand
///////////////////////////////////////////////////////////////////////////////

typedef enum {} curandOrdering_t;
typedef hiprandRngType curandRngType_t;
typedef hiprandStatus_t curandStatus_t;

typedef hiprandGenerator_t curandGenerator_t;

curandRngType_t convert_hiprandRngType(curandRngType_t t) {
    switch(static_cast<int>(t)) {
    case 100: return HIPRAND_RNG_PSEUDO_DEFAULT;
    case 101: return HIPRAND_RNG_PSEUDO_XORWOW;
    case 121: return HIPRAND_RNG_PSEUDO_MRG32K3A;
    case 141: return HIPRAND_RNG_PSEUDO_MTGP32;
    case 142: return HIPRAND_RNG_PSEUDO_MT19937;
    case 161: return HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
    case 200: return HIPRAND_RNG_QUASI_DEFAULT;
    case 201: return HIPRAND_RNG_QUASI_SOBOL32;
    case 202: return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
    case 203: return HIPRAND_RNG_QUASI_SOBOL64;
    case 204: return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    }
    return HIPRAND_RNG_TEST;
}

// curandGenerator_t
curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) {
    rng_type = convert_hiprandRngType(rng_type);
    return hiprandCreateGenerator(generator, rng_type);
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    return hiprandDestroyGenerator(generator);
}

curandStatus_t curandGetVersion(int *version) {
    return hiprandGetVersion(version);
}


// Stream
curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
    return hiprandSetStream(generator, stream);
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) {
    return hiprandSetPseudoRandomGeneratorSeed(generator, seed);
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
    return hiprandSetGeneratorOffset(generator, offset);
}

curandStatus_t curandSetGeneratorOrdering(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}


// Generation functions
curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int *output_data, size_t n) {
    return hiprandGenerate(generator, output_data, n);
}

curandStatus_t curandGenerateLongLong(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *output_data, size_t n) {
    return hiprandGenerateUniform(generator, output_data, n);
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *output_data, size_t n) {
    return hiprandGenerateUniformDouble(generator, output_data, n);
}

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *output_data, size_t n, float mean, float stddev) {
    return hiprandGenerateNormal(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *output_data, size_t n, double mean, double stddev) {
    return hiprandGenerateNormalDouble(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *output_data, size_t n, float mean, float stddev) {
    return hiprandGenerateLogNormal(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *output_data, size_t n, double mean, double stddev) {
    return hiprandGenerateLogNormalDouble(generator, output_data, n, mean, stddev);
}

curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int *output_data, size_t n, double lambda) {
    return hiprandGeneratePoisson(generator, output_data, n, lambda);
}

///////////////////////////////////////////////////////////////////////////////
// cuda_profiler_api.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} cudaOutputMode_t;

cudaError_t cudaProfilerInitialize(...) {
  return cudaSuccess;
}

cudaError_t cudaProfilerStart() {
  return hipProfilerStart();
}

cudaError_t cudaProfilerStop() {
  return hipProfilerStop();
}

} // extern "C"

#elif !defined(CUPY_NO_CUDA)
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#ifndef CUPY_NO_NVTX
#include <nvToolsExt.h>
#endif // #ifndef CUPY_NO_NVTX

extern "C" {

bool hip_environment = false;

#if CUDA_VERSION < 8000
#if CUDA_VERSION >= 7050
typedef cublasDataType_t cudaDataType;
#else
enum cudaDataType_t {};
typedef enum cudaDataType_t cudaDataType;
#endif // #if CUDA_VERSION >= 7050
#endif // #if CUDA_VERSION < 8000


#if CUDA_VERSION < 7050
cublasStatus_t cublasSgemmEx(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

#endif // #if CUDA_VERSION < 7050


#if CUDA_VERSION < 8000

enum cudaMemoryAdvise {};

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                 int dstDevice, cudaStream_t stream) {
    return cudaErrorUnknown;
}

cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                          enum cudaMemoryAdvise advice, int device) {
    return cudaErrorUnknown;
}

typedef enum {} cublasGemmAlgo_t;

cublasStatus_t cublasSgemmStridedBatched(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasDgemmStridedBatched(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasCgemmStridedBatched(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasZgemmStridedBatched(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGemmEx(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

#endif // #if CUDA_VERSION < 8000


#if CUDA_VERSION < 9000

typedef enum {} cublasMath_t;

cublasStatus_t cublasSetMathMode(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasGetMathMode(...) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

#endif // #if CUDA_VERSION < 9000

} // extern "C"

#else // #ifndef CUPY_NO_CUDA

#define CUDA_VERSION 0

extern "C" {

bool hip_environment = false;

///////////////////////////////////////////////////////////////////////////////
// cuda.h
///////////////////////////////////////////////////////////////////////////////

typedef int CUdevice;
typedef enum {
    CUDA_SUCCESS = 0,
} CUresult;
enum CUjit_option {};
enum CUjitInputType {};


typedef void* CUdeviceptr;
struct CUctx_st;
struct CUevent_st;
struct CUfunc_st;
struct CUmod_st;
struct CUstream_st;
struct CUlinkState_st;


typedef struct CUctx_st* CUcontext;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUfunc_st* CUfunction;
typedef struct CUmod_st* CUmodule;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUlinkState_st* CUlinkState;

// Error handling
CUresult cuGetErrorName(...) {
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(...) {
    return CUDA_SUCCESS;
}

// Primary context management
CUresult cuDevicePrimaryCtxRelease(...) {
    return CUDA_SUCCESS;
}

// Context management
CUresult cuCtxGetCurrent(...) {
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(...) {
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(...) {
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(...) {
    return CUDA_SUCCESS;
}


// Module load and kernel execution
CUresult cuLinkCreate (...) {
    return CUDA_SUCCESS;
}

CUresult cuLinkAddData(...) {
    return CUDA_SUCCESS;
}

CUresult cuLinkComplete(...) {
    return CUDA_SUCCESS;
}

CUresult cuLinkDestroy(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleLoad(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(...) {
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(...) {
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(...) {
    return CUDA_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////
// cuda_runtime.h
///////////////////////////////////////////////////////////////////////////////

enum {
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
};

typedef enum {
    cudaSuccess = 0,
} cudaError_t;
typedef enum {} cudaDataType;
enum cudaDeviceAttr {};
enum cudaMemoryAdvise {};
enum cudaMemcpyKind {};


typedef void (*cudaStreamCallback_t)(
    cudaStream_t stream, cudaError_t status, void* userData);


struct cudaPointerAttributes{
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    int memoryType;
};


// Error handling
const char* cudaGetErrorName(...) {
    return NULL;
}

const char* cudaGetErrorString(...) {
    return NULL;
}


// Initialization
cudaError_t cudaDriverGetVersion(...) {
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(...) {
    return cudaSuccess;
}


// CUdevice operations
cudaError_t cudaGetDevice(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(...) {
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(...) {
    return cudaSuccess;
}

cudaError_t cudaSetDevice(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(...) {
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(...) {
    return cudaSuccess;
}


// Memory management
cudaError_t cudaMalloc(...) {
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(...) {
    return cudaSuccess;
}

cudaError_t cudaMallocManaged(...) {
    return cudaSuccess;
}

int cudaFree(...) {
    return cudaSuccess;
}

cudaError_t cudaFreeHost(...) {
    return cudaSuccess;
}

int cudaMemGetInfo(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpy(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeer(...) {
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeerAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemset(...) {
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(...) {
    return cudaSuccess;
}

cudaError_t cudaMemAdvise(...) {
    return cudaSuccess;
}

cudaError_t cudaMemPrefetchAsync(...) {
    return cudaSuccess;
}


cudaError_t cudaPointerGetAttributes(...) {
    return cudaSuccess;
}


// Stream and Event
cudaError_t cudaStreamCreate(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(...) {
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(...) {
    return cudaSuccess;
}

cudaError_t cudaEventCreate(...) {
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(...) {
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(...) {
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(...) {
    return cudaSuccess;
}

cudaError_t cudaEventQuery(...) {
    return cudaSuccess;
}

cudaError_t cudaEventRecord(...) {
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(...) {
    return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// cuComplex.h
///////////////////////////////////////////////////////////////////////////////

#include "cupy_cuComplex.h"

///////////////////////////////////////////////////////////////////////////////
// cublas_v2.h
///////////////////////////////////////////////////////////////////////////////

typedef void* cublasHandle_t;

typedef enum {} cublasDiagType_t;
typedef enum {} cublasFillMode_t;
typedef enum {} cublasOperation_t;
typedef enum {} cublasPointerMode_t;
typedef enum {} cublasSideMode_t;
typedef enum {} cublasGemmAlgo_t;
typedef enum {} cublasMath_t;
typedef enum {
    CUBLAS_STATUS_SUCCESS=0,
} cublasStatus_t;


// Context
cublasStatus_t cublasCreate(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// Stream
cublasStatus_t cublasSetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// Math Mode
cublasStatus_t cublasSetMathMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotu(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotu(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}


// BLAS Level 2
cublasStatus_t cublasSgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasCgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

int cublasCgeru(...) {
    return 0;
}

int cublasCgerc(...) {
    return 0;
}

int cublasZgeru(...) {
    return 0;
}

int cublasZgerc(...) {
    return 0;
}

// BLAS Level 3
cublasStatus_t cublasSgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasCgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}


// BLAS extension
cublasStatus_t cublasSgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdgmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////
// curand.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} curandOrdering_t;
typedef enum {} curandRngType_t;
typedef enum {
    CURAND_STATUS_SUCCESS = 0,
} curandStatus_t;

typedef void* curandGenerator_t;


// curandGenerator_t
curandStatus_t curandCreateGenerator(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetVersion(...) {
    return CURAND_STATUS_SUCCESS;
}


// Stream
curandStatus_t curandSetStream(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(...) {
    return CURAND_STATUS_SUCCESS;
}


// Generation functions
curandStatus_t curandGenerate(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLongLong(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniform(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormal(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormalDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormal(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormalDouble(...) {
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGeneratePoisson(...) {
    return CURAND_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// cuda_profiler_api.h
///////////////////////////////////////////////////////////////////////////////

typedef enum {} cudaOutputMode_t;

cudaError_t cudaProfilerInitialize(...) {
  return cudaSuccess;
}

cudaError_t cudaProfilerStart() {
  return cudaSuccess;
}

cudaError_t cudaProfilerStop() {
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// nvToolsExt.h
///////////////////////////////////////////////////////////////////////////////

#define NVTX_VERSION 1

typedef enum nvtxColorType_t
{
    NVTX_COLOR_UNKNOWN  = 0,
    NVTX_COLOR_ARGB     = 1
} nvtxColorType_t;

typedef enum nvtxMessageType_t
{
    NVTX_MESSAGE_UNKNOWN          = 0,
    NVTX_MESSAGE_TYPE_ASCII       = 1,
    NVTX_MESSAGE_TYPE_UNICODE     = 2,
} nvtxMessageType_t;

typedef union nvtxMessageValue_t
{
    const char* ascii;
    const wchar_t* unicode;
} nvtxMessageValue_t;

typedef struct nvtxEventAttributes_v1
{
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t colorType;
    uint32_t color;
    int32_t payloadType;
    int32_t reserved0;
    union payload_t
    {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
    } payload;
    int32_t messageType;
    nvtxMessageValue_t message;
} nvtxEventAttributes_v1;

typedef nvtxEventAttributes_v1 nvtxEventAttributes_t;

void nvtxMarkA(...) {
}

void nvtxMarkEx(...) {
}

int nvtxRangePushA(...) {
    return 0;
}

int nvtxRangePushEx(...) {
    return 0;
}

int nvtxRangePop() {
    return 0;
}

uint64_t nvtxRangeStartEx(...) {
    return 0;
}

void nvtxRangeEnd(...) {
}

} // extern "C"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDA_H
