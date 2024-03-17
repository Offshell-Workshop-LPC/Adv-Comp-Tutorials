#include "cuda_runtime.h"
#include <cstdio>
#include <cuda/std/cassert>



#define __CUDA_HOST_DEVICE__ __host__ __device__
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


template<typename T> struct TestClass{
  T x;

  __CUDA_HOST_DEVICE__ TestClass(T const& x) : x(x){}
  __CUDA_HOST_DEVICE__ operator T() const{ return x; }
};
template<> struct TestClass<float>{
  float x;

  __CUDA_HOST_DEVICE__ TestClass(float const& x) : x(x+1e-3f){}
  __CUDA_HOST_DEVICE__ operator float() const{ return x; }
};
template<typename T> struct TestClass<TestClass<T>>{
  T x;

  __CUDA_HOST_DEVICE__ TestClass(TestClass<T> const& x) : x(-x){}
  __CUDA_HOST_DEVICE__ operator T() const{ return x; }
};

template<typename T> __CUDA_HOST_DEVICE__ void print_value(T const& x){
  printf("Value is %f.\n", static_cast<double>(x));
}
template<> __CUDA_HOST_DEVICE__ void print_value(float const& x){
  printf("Twice the value of the float %f is %f.\n", x, 2*x);
}


template<typename T> __CUDA_GLOBAL__ void print_value_kernel(unsigned int n, T* data){
  unsigned int const i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) print_value(data[i]);
}


int main(){
  // This is the regular printout from CPU, pretty much unmodified.
  print_value(5); // int
  print_value(2.3); // double
  print_value(5.1f); // float

  // The following lines will call print_value(TestClass<T> const& x) even if operator T() is defined!
  print_value(TestClass<double>(7.25)); // struct initialization with a double type argument
  print_value(TestClass<float>(1.22f)); // struct initialization with a float type argument

  // This is the correct way to force the call print_value(float const& x).
  print_value((float) TestClass<float>(1.22f));

  TestClass<double> cx(87);
  print_value(TestClass<TestClass<double>>(cx));

  /*
  GPU printouts:
  Notice here the execution order and concurrency as well.
  Kernel calls are asynchronous with the CPU, so you need to orchestrate how you wait for the GPU.
  In this example, call (5) running on the CPU does not have to wait for the GPU call sequence (3+4).
  That is why when you call cudaMemcpy, you need to synchronize the GPU device.
  Otherwise, for a large array of data, you could call free(h_ccx) before cudaMemcpy finishes.
  */
  constexpr unsigned int n = 100;
  constexpr unsigned long long int total_size = n*sizeof(TestClass<TestClass<double>>);
  using T = TestClass<TestClass<double>>;

  // Here is one way way to make an array of objects on the host,
  // keeping the memory allocation method in the same style
  // as the allocation of memory on the device. (1)
  T* h_ccx = nullptr;
  h_ccx = (T*) malloc(total_size);
  for (unsigned int i = 0; i < n; ++i) new (h_ccx+i) T(static_cast<double>(i+1)); // Placement new
  // Allocate GPU memory (2)
  T* d_ccx = nullptr;
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMalloc(&d_ccx, total_size));
  // Transfer of h_ccx -> d_ccx (3)
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpy(d_ccx, h_ccx, total_size, cudaMemcpyHostToDevice));
  // Need to sync the GPU before we eventually free the memory
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaDeviceSynchronize());
  // Kernel call (4)
  print_value_kernel<<<1, n>>>(n, d_ccx);
  // Can already free the host memory at this point (5)
  free(h_ccx);
  // Free the GPU memory (6)
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFree(d_ccx));

  return 0;
}
