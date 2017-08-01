#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void arrayset_kernel(int n, float value, float* output){
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = value;
}
int DLGpuArraySet(DLArrayHandle arr, float value){
    float *output_data = (float *)arr->data;
    int n = 1;
    for (int i=0; i<arr->ndim; i++) n *= arr->shape[i];
    int size = 1024;
    dim3  threads(size);
    dim3  blocks((n + size - 1)/size);
    arrayset_kernel<<<blocks, threads>>>(n, value, output_data);
    return 0;
}

__global__ void broadcastto_kernel(int m, int n, const float* input_data, float* output_data){

    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if(idx >=  m){
        return;
    }
    for (int i=0; i<n; i++) output_data[i * m + idx] = input_data[idx];
}


int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
    int input_size = 1;
    int output_size = 1;
    for (int i=0; i<input->ndim; i++) input_size*=input->shape[i];
    for (int i=0; i<output->ndim; i++) output_size*=output->shape[i];
    assert(output_size > input_size);
    assert(output_size % input_size == 0);
    const float * input_data = (const float *)input->data;
    float * output_data = (float *)output->data;
    int size = 1024;
    dim3 threads(size);
    dim3 blocks((input_size + size -1) / size);
    int batch_size = output_size/input_size;
    broadcastto_kernel<<<blocks, threads>>>(input_size, batch_size, input_data, output_data);
  return 0;
}

__global__ void reducesumaxiszero_kernel(int output_size, int nrow, const float* input_data, float* output_data){
    
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    output_data[idx] = 0.0;
    for(int i=0; i<nrow; i++) output_data[idx] += input_data[i*output_size + idx];
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
    int input_size = 1;
    int output_size = 1;

    for (int i=0; i<input->ndim; i++) input_size*=input->shape[i];
    for (int i=0; i<output->ndim; i++) output_size*=output->shape[i];

    int m = input->shape[0];
    const float* input_data = (const float *) input->data;
    float * output_data = (float *)output->data;
    int size = 1024;
    dim3 threads(size);
    dim3 blocks((output_size + size - 1)/size);
    reducesumaxiszero_kernel<<<blocks, threads>>>(output_size, m, input_data, output_data);
  return 0;
}

__global__ void matrixelementwiseadd_kernel(int n, const float* A, 
                                            const float* B, float* C){
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if( idx >= n ) return;
    C[idx] = A[idx] + B[idx];
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
    assert(matA->ndim == matB->ndim);
    int n = 1;
    for (int i=0; i<matA->ndim; i++){
        if(matA->shape[i] == matB->shape[i]){
            n *= matA->shape[i];
        }
    }
    int size = 1024;
    dim3 threads(size);
    dim3 blocks((n + size - 1)/size);

    const float* data_A  = (const float*)matA->data;
    const float* data_B  = (const float*)matB->data;
    float* data_C  = (float*)output->data;
    matrixelementwiseadd_kernel<<<blocks, threads>>>(n, data_A, data_B, data_C);
  return 0;
}

__global__ void matrixelementwiseaddbyconst_kernel(int n, const float* src, float val, float* dst){

    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if( idx >= n ) return;
    dst[idx] = src[idx] + val;
}
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
    int n = 1;
    for (int i=0; i<input->ndim; i++){
            n *= input->shape[i];
    }
    int size = 1024;
    dim3 threads(size);
    dim3 blocks((n + size - 1)/size);

    const float* data_A  = (const float*)input->data;
    float* data_C  = (float*)output->data;
    matrixelementwiseaddbyconst_kernel<<<blocks, threads>>>(n, data_A, val, data_C);
    return 0;
}

__global__ void matrixelementwisemultiply_kernel(int n, 
                                                const float* src_A,
                                                const float* src_B,
                                                float* dst){
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if( idx >= n ) return;
    dst[idx] = src_A[idx] * src_B[idx];
}
int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
    int n = 1;
    int m = 1;
    for (int i=0; i<matA->ndim; i++){
            m *= matA->shape[i];
    }
    for (int i=0; i<matB->ndim; i++){
            n *= matB->shape[i];
    }
    assert(m == n);
    int size = 1024;
    dim3 threads(size);
    dim3 blocks((n + size - 1)/size);

    const float* data_A  = (const float*)matA->data;
    const float* data_B  = (const float*)matB->data;
    float* data_C  = (float*)output->data;
    matrixelementwisemultiply_kernel<<<blocks, threads>>>(n, data_A, data_B, data_C);
  return 0;
}
__global__ void matrix_multiply_by_const_kernel(int m, const float* src_A, 
                                                float* dst, float val){

    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if( idx >= m ) return;
    dst[idx] = src_A[idx] * val;
}
int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
    int m = 1;
    for (int i=0; i<input->ndim; i++){
            m *= input->shape[i];
    }
    int size = 1024;
    dim3 threads(size);
    dim3 blocks((m + size - 1)/size);

    const float* data_A  = (const float*)input->data;
    float* data_C  = (float*)output->data;
    matrix_multiply_by_const_kernel<<<blocks, threads>>>(m, data_A, data_C, val);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);
    cudaThreadSynchronize();
    int nrow_A = matA->shape[0];
    int ncol_A = matA->shape[1];
    int nrow_B = matB->shape[0];
    int ncol_B = matB->shape[1];
    if (transposeA) std::swap(nrow_A, ncol_A);
    if (transposeB) std::swap(nrow_B, ncol_B);
    cublasOperation_t trans_A = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_B = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const float *input_data_A = (const float *)matA->data;
    const float *input_data_B = (const float *)matB->data;
    float *output_data = (float *)matC->data;
    assert(nrow_A == matC->shape[0] && ncol_B == matC->shape[1]);
    assert(ncol_A == nrow_B);

    float a = 1, b = 0;

    cublasSgemm(handle,
                trans_B,
                trans_A,
                ncol_B,
                nrow_A,
                nrow_B,
                &a,
                input_data_B,   
                transposeB ? nrow_B : ncol_B,
                input_data_A,
                transposeA ? nrow_A : ncol_A,
                &b,
                output_data,
                ncol_B
                );
    cudaThreadSynchronize();
    return 0;
}
__global__ void gpu_relu_kernel(int n, const float* in, float* out){
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (idx >=n )return;
    out[idx] = in[idx]>0 ? in[idx]:0.0;
}
int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
    int m = 1;
    int n = 1;
    for (int i=0; i<input->ndim; i++){
            m *= input->shape[i];
    }
    for (int i=0; i<output->ndim; i++){
            n *= output->shape[i];
    }
    assert(m == n);
    int size = 1024;
    dim3 threads(size);
    dim3 blocks((m + size - 1)/size);

    const float* data_A  = (const float*)input->data;
    float* data_C  = (float*)output->data;
    gpu_relu_kernel<<<blocks, threads>>>(m, data_A, data_C);
  return 0;
}

__global__  void gpu_relu_grad_kernel(int m, const float* data_A, 
                                const float* data_B, float* data_C)
{
    int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (idx >=m )return;
    data_C[idx] = data_A[idx]>0 ? data_B[idx]:0.0;
}
int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
    int m = 1;
    int n = 1;
    int p = 1;
    for (int i=0; i<input->ndim; i++){
            m *= input->shape[i];
    }
    for (int i=0; i<in_grad->ndim; i++){
            n *= in_grad->shape[i];
    }
    for (int i=0; i<output->ndim; i++){
            p *= output->shape[i];
    }
    assert(m == n && m == p);
    int size = 1024;
    dim3 threads(size);
    dim3 blocks((m + size - 1)/size);

    const float* data_A  = (const float*)input->data;
    const float* data_B  = (const float*)in_grad->data;
    float* data_C  = (float*)output->data;
    gpu_relu_grad_kernel<<<blocks, threads>>>(m, data_A, data_B, data_C);
  return 0;
}

__global__ void matrix_softmax(int nrow, int ncol, const float* input, float* output){
    
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input += y * ncol;
  output += y * ncol;
  float maxval = *input;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input[x] - maxval);
  }
  // Compute per-row loss.
  for (int x = 0; x < ncol; ++x) {
    output[x] = exp(input[x] - maxval) / sum;
  }
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  int nrow = input->shape[0];
  assert(nrow <= 1024 * 4);
  int ncol = input -> shape[1];
  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  matrix_softmax<<<1, threads>>>(
      nrow, ncol, input_data, output_data);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
