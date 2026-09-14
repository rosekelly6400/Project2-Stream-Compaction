#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void formatForNonPowOf2(int nOriginal, int nPowerOf2, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= nPowerOf2) return;
            int zeroBufferSize = nPowerOf2 - nOriginal;

            if (idx < zeroBufferSize)
            {
                odata[idx] = 0;
            }
            else {
                odata[idx] = idata[idx- zeroBufferSize];
            }

        }

        __global__ void postScanFormat(int nOriginal, int nPowerOf2, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= nPowerOf2) return;
            int zeroBufferSize = nPowerOf2 - nOriginal;

            if (idx == 0)
            {
                odata[0] = 0;
            }
            else {
                odata[idx] = idata[idx + (zeroBufferSize-1)];
            }

        }

        __global__ void scan_GPU_naive(int n, int iteration, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            int spacing = std::pow(2 , iteration - 1);
            if (idx >= spacing)
            {
                odata[idx] = idata[idx - spacing] + idata[idx];
            }
            else {
                odata[idx] = idata[idx];
            }
        }
        // TODO: __global__

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int blockSize = 32;
            // include zero buffer for non power of 2 n
            int log2n = ilog2ceil(n);
            int dataLength = std::pow(2, log2n);

            int sizeInBytes = dataLength * sizeof(int);

            int* dev_in;
            int* dev_out;

            cudaMalloc((void**)&dev_in, sizeInBytes);
            checkCUDAError("cudaMalloc dev_in failed!");

            cudaMalloc((void**)&dev_out, sizeInBytes);
            checkCUDAError("cudaMalloc dev_out failed!");

            cudaMemcpy(dev_in, idata, sizeInBytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // change number of threads to be ceiling of log2(n)

            dim3 fullBlocksPerGrid((dataLength + blockSize - 1) / blockSize);

            cudaDeviceSynchronize();

            // add zeros to beginning of buffer if n is not power of 2
            if (dataLength != n)
            {
                formatForNonPowOf2 << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_out, dev_in);
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }

            // FIGURE OUT: why does it pass for log2n+1 instead of just log2n+1
            // ANSWER: its because I was still switching in and out after the last iteration, so I was returning the 2nd to last iteration's out as the final out
            // sum
            for (int i = 1; i <= log2n; i++)
            {
                scan_GPU_naive << <fullBlocksPerGrid, blockSize >> > (dataLength, i, dev_out, dev_in);
                if (i != log2n)
                {
                    int* temp = dev_in;
                    dev_in = dev_out;
                    dev_out = temp;
                }
            }

            // post process to make it an exclusive scan and remove trailing zeros from non power of 2
            postScanFormat << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_in, dev_out);
            int* temp = dev_in;
            dev_in = dev_out;
            dev_out = temp;

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeInBytes, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
        }

        // This version can handle arrays only as large as can be processed by a single thread block running on one multiprocessor of a GPU.
        __global__ void scan_GPU_naive_shared(int n, int log2n, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            extern __shared__ int shared_idata[];
            shared_idata[idx] = idata[idx];
            __syncthreads();

            for (int i = 1; i <= log2n; i++)
            {
                int spacing = std::pow(2, i - 1);
                if (idx >= spacing)
                {
                    shared_idata[idx] = shared_idata[idx - spacing] + shared_idata[idx];
                }
                // CHANGE ABOVE SO IT RUNS HALF THE THREADS ??
                __syncthreads();
            }
            odata[idx] = shared_idata[idx];
        }

        void scanShared(int n, int* odata, const int* idata) {
            // TODO
            // include zero buffer for non power of 2 n
            int blockSize = 1024;
            int log2n = ilog2ceil(n);
            int dataLength = std::pow(2, log2n);
            if (n < blockSize) {
                blockSize = n;
            }
            else {
                checkCUDAError("Array length is too big for shared mem approach!");
            }

            int sizeInBytes = dataLength * sizeof(int);

            int* dev_in;
            int* dev_out;

            cudaMalloc((void**)&dev_in, sizeInBytes);
            checkCUDAError("cudaMalloc dev_in failed!");

            cudaMalloc((void**)&dev_out, sizeInBytes);
            checkCUDAError("cudaMalloc dev_out failed!");

            cudaMemcpy(dev_in, idata, sizeInBytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // change number of threads to be ceiling of log2(n)

            dim3 fullBlocksPerGrid((dataLength + blockSize - 1) / blockSize);

            cudaDeviceSynchronize();

            // add zeros to beginning of buffer if n is not power of 2
            if (dataLength != n)
            {
                formatForNonPowOf2 << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_out, dev_in);
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }

            // scan using shared mem
            scan_GPU_naive_shared << <fullBlocksPerGrid, blockSize, sizeInBytes >> > (dataLength, log2n, dev_out, dev_in);

            // post process to make it an exclusive scan and remove trailing zeros from non power of 2
            postScanFormat << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_in, dev_out);
            int* temp = dev_in;
            dev_in = dev_out;
            dev_out = temp;

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeInBytes, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
        }
    }
}
