#ifndef IVYCUDAFLAGS_H
#define IVYCUDAFLAGS_H


#ifdef __USE_CUDA__

#define __CUDA_HOST__ __host__
#define __CUDA_DEVICE__ __device__
#define __CUDA_GLOBAL__ __global__
#define __CUDA_HOST_DEVICE__ __host__ __device__
#define __CUDA_DEVICE_HOST__ __device__ __host__

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#define __CUDA_DEVICE_CODE__
#endif

#define GlobalCudaStreamRaw cudaStreamLegacy

#define __CUDA_CHECK_SUCCESS__(CALL) ((CALL) == cudaSuccess)

#define __CUDA_MANAGED__ __managed__
#define __CUDA_CONSTANT__ __constant__
#define __CUDA_SHARED__ __shared__

#define __RESTRICT__ __restrict__
#define __INLINE_FCN_FORCE__ __forceinline__
#define __INLINE_FCN_RELAXED__ inline
#define __INLINE_VAR__ inline

#define __STATIC_CAST__(TYPE, PTR) static_cast<TYPE>(PTR)
#define __CONST_CAST__(TYPE, PTR) const_cast<TYPE>(PTR)
#define __REINTERPRET_CAST__(TYPE, PTR) reinterpret_cast<TYPE>(PTR)
#ifdef __CUDA_DEVICE_CODE__
#define __DYNAMIC_CAST__(TYPE, PTR) static_cast<TYPE>(PTR)
#else
#define __DYNAMIC_CAST__(TYPE, PTR) dynamic_cast<TYPE>(PTR)
#endif

#define __CUDA_THREADFENCE_BLOCK__ __threadfence_block();
#define __CUDA_THREADFENCE__ __threadfence();
#define __CUDA_THREADFENCE_SYSTEM__ __threadfence_system();
#define __CUDA_SYNCTHREADS__ __syncthreads();
#define __CUDA_SYNCTHREADS_COUNT__(PREDICATE) __syncthreads_count(PREDICATE);
#define __CUDA_SYNCTHREADS_AND__(PREDICATE) __syncthreads_and(PREDICATE);
#define __CUDA_SYNCTHREADS_OR__(PREDICATE) __syncthreads_or(PREDICATE);
#define __CUDA_SYNCWARP__(PREDICATE) __syncwarp(MASK);

#else

#define __CUDA_HOST__
#define __CUDA_DEVICE_
#define __CUDA_GLOBAL__
#define __CUDA_HOST_DEVICE__ __CUDA_HOST__ __CUDA_DEVICE_
#define __CUDA_DEVICE_HOST__ __CUDA_DEVICE_ __CUDA_HOST__

#define GlobalCudaStreamRaw

#define __CUDA_MANAGED__
#define __CUDA_CONSTANT__
#define __CUDA_SHARED__

#define __RESTRICT__ restrict
#define __INLINE_FCN_FORCE__ inline
#define __INLINE_FCN_RELAXED__ inline
#define __INLINE_VAR__ inline

#define __STATIC_CAST__(TYPE, PTR) static_cast<TYPE>(PTR)
#define __DYNAMIC_CAST__(TYPE, PTR) dynamic_cast<TYPE>(PTR)
#define __CONST_CAST__(TYPE, PTR) const_cast<TYPE>(PTR)
#define __REINTERPRET_CAST__(TYPE, PTR) reinterpret_cast<TYPE>(PTR)

#define __CUDA_THREADFENCE_BLOCK__
#define __CUDA_THREADFENCE__
#define __CUDA_THREADFENCE_SYSTEM__
#define __CUDA_SYNCTHREADS__
#define __CUDA_SYNCTHREADS_COUNT__(PREDICATE)
#define __CUDA_SYNCTHREADS_AND__(PREDICATE)
#define __CUDA_SYNCTHREADS_OR__(PREDICATE)
#define __CUDA_SYNCWARP__(PREDICATE)

#endif


#endif
