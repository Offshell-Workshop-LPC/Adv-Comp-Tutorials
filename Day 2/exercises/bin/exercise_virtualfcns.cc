#include "cuda_runtime.h"
#include <cstdio>
#include <cuda/std/cassert>



#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#define __CUDA_DEVICE_CODE__
#endif
#define __CUDA_HOST_DEVICE__ __host__ __device__
#define __CUDA_GLOBAL__ __global__
#define __CUDA_HOST__ __host__
#define __CUDA_DEVICE__ __device__
// Fancy way to check an error in a CUDA function call
#define __CUDA_CHECK_OR_EXIT_WITH_ERROR__(CALL) \
{ \
  auto __cuda_error_code__ = CALL; \
  if (__cuda_error_code__ != cudaSuccess){ \
    printf("*** CUDA call ***\n*** '%s' ***\n*** at ***\n*** %s::%d ***\n*** failed with error '%s'. ***\n", #CALL, __FILE__, __LINE__, cudaGetErrorString(__cuda_error_code__)); \
    assert(false); \
  } \
}


struct Base{
  virtual __CUDA_DEVICE__ void print() const = 0;
};
struct Derived : Base{
  __CUDA_DEVICE__ void print() const{
    printf("Derived::print() called.\n");
  }
};


__CUDA_GLOBAL__ void test_print(Derived* obj){
  unsigned int const i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < 1) obj->print();
}
__CUDA_GLOBAL__ void test_print_v2(int){
  unsigned int const i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < 1) Derived().print();
}


int main(){
  constexpr unsigned long long int total_size = sizeof(Derived);

  Derived* h_ptr = (Derived*) malloc(total_size);
  h_ptr = new (h_ptr) Derived();

  Derived* d_ptr = nullptr;
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMalloc(&d_ptr, total_size));
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpy(d_ptr, h_ptr, total_size, cudaMemcpyHostToDevice));
  printf("test_print:\n");
  test_print<<<1, 1>>>(d_ptr);
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaDeviceSynchronize());
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFree(d_ptr));
  free(h_ptr);

  printf("test_print_v2:\n");
  test_print_v2<<<1, 1>>>(2);
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaDeviceSynchronize());

  return 0;
}
