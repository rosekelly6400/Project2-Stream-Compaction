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
            int zeroBufferSize = nPowerOf2 - nOriginal;
            if ((idx + (zeroBufferSize - 1)) >= nPowerOf2 || (idx + (zeroBufferSize - 1)) < 0 || idx >= nPowerOf2) return;

            odata[idx] = idata[idx + (zeroBufferSize-1)];
        }

        __global__ void setFirstElementToZero(int n, int* odata)
        {
            odata[0] = 0;
        }

        __global__ void scan_GPU_naive(int n, int iteration, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            int spacing = 1 << (iteration - 1);
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
            int blockSize = 1024;
            // include zero buffer for non power of 2 n
            int log2n = ilog2ceil(n);
            int dataLength = 1 << log2n;

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

            // FIGURE OUT: why does it pass for log2n+1 instead of just log2n
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

            //// post process to make it an exclusive scan and remove trailing zeros from non power of 2
            postScanFormat << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_in, dev_out);
            int* temp = dev_in;
            dev_in = dev_out;
            dev_out = temp;
            setFirstElementToZero << <1, 1 >> > (dataLength, dev_out);

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
                int spacing = 1 << (i - 1);
                if (idx >= spacing)
                {
                    shared_idata[idx] = shared_idata[idx - spacing] + shared_idata[idx];
                }
                // CHANGE ABOVE SO IT RUNS HALF THE THREADS ??
                __syncthreads();
            }
            odata[idx] = shared_idata[idx];
        }


        __global__ void scan_GPU_naive_shared_multiBlock(int n, int log2n, int numBlocks, int blockSize, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned sharedIdx = threadIdx.x;
            if (idx >= n || sharedIdx >= blockSize) return;
            int outHalfIndex = 0;
            int inHalfIndex = 1;
            extern __shared__ int shared_idata[];
            shared_idata[outHalfIndex* blockSize + sharedIdx] = idata[idx];
            //shared_idata[inHalfIndex*n + sharedIdx] = 0;
            __syncthreads();

            for (int i = 1; i <= log2n; i++)
            {
                outHalfIndex = 1 - outHalfIndex;
                inHalfIndex = 1 - outHalfIndex;
                int spacing = 1 << (i - 1);
                if (sharedIdx >= spacing && sharedIdx < blockSize)
                {
                    // TRY PING PONGING INSTEAD OF THIS???
                    shared_idata[outHalfIndex * blockSize + sharedIdx] = shared_idata[inHalfIndex * blockSize + sharedIdx - spacing] + shared_idata[inHalfIndex * blockSize + sharedIdx];
                }
                else
                {
                    shared_idata[outHalfIndex * blockSize + sharedIdx] = shared_idata[inHalfIndex * blockSize + sharedIdx];
                }
                // CHANGE ABOVE SO IT RUNS HALF THE THREADS ??
                __syncthreads();
            }
            odata[idx] = shared_idata[outHalfIndex * blockSize + sharedIdx];
        }

        __global__ void copyBlockSumsToOutData(int numBlocks, int blockSize, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numBlocks) return;

            odata[idx] = idata[idx];
        }

        void scanShared(int n, int* odata, const int* idata) {
            // TODO
            // include zero buffer for non power of 2 n
            int blockSize = 1024;
            int log2n = ilog2ceil(n);
            int dataLength = 1 << log2n;
            if (dataLength < blockSize) {
                blockSize = dataLength;
            }

            int sizeInBytes = dataLength * sizeof(int);

            int numBlocks = (dataLength + blockSize - 1) / blockSize;

            int* dev_in;
            int* dev_out;
            int* dev_blockSums;
            int* dev_blockSumsScanned;

            cudaMalloc((void**)&dev_in, sizeInBytes);
            checkCUDAError("cudaMalloc dev_in failed!");
            cudaMalloc((void**)&dev_out, sizeInBytes);
            checkCUDAError("cudaMalloc dev_out failed!");
            cudaMalloc((void**)&dev_blockSums, numBlocks *sizeof(int));
            checkCUDAError("cudaMalloc dev_blockSums failed!");
            cudaMalloc((void**)&dev_blockSumsScanned, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc dev_blockSums failed!");

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

            // 1. scan array as blocks of blockSize
           /* for (int i = 0; i < numBlocks; i++)
            {
                int sharedMemSizeInBytes = blockSize * sizeof(int);
                int log2BlockSize = ilog2ceil(blockSize);
                scan_GPU_naive_shared << <fullBlocksPerGrid, blockSize, sharedMemSizeInBytes >> > (blockSize, log2BlockSize, dev_out + (i * blockSize), dev_in + (i * blockSize));
            }*/
            int sharedMemSizeInBytes = blockSize * sizeof(int);
            int log2BlockSize = ilog2ceil(blockSize);
            cudaDeviceSynchronize();
            scan_GPU_naive_shared_multiBlock << <fullBlocksPerGrid, blockSize, sharedMemSizeInBytes*2 >> > (dataLength, log2BlockSize, numBlocks, blockSize, dev_out, dev_in);
            cudaDeviceSynchronize();
            // 2. Write total sum of each block into a new array
            dim3 blockSumsBlocksPerGrid((numBlocks + blockSize - 1) / blockSize);
            StreamCompaction::Common::copyBlockSums << <blockSumsBlocksPerGrid, blockSize >> > (numBlocks, blockSize, dev_blockSums, dev_out);
            // 3. Exclusive scan that array (or inclusive scan and ignore last element and add to next section)
            int blockSumsSharedMemSizeInBytes = numBlocks * sizeof(int);
            int log2NumBlocks = ilog2ceil(numBlocks);
            for (int i = 1; i <= log2NumBlocks; i++)
            {
                scan_GPU_naive << <fullBlocksPerGrid, blockSize >> > (numBlocks, i, dev_blockSumsScanned, dev_blockSums);
                if (i != log2NumBlocks)
                {
                    int* temp = dev_blockSums;
                    dev_blockSums = dev_blockSumsScanned;
                    dev_blockSumsScanned = temp;
                }
            }
            // 4. Add each element back to its section (or next section if inclusive scan)
            StreamCompaction::Common::addBlockSumsBack << <fullBlocksPerGrid, blockSize >> > (dataLength, numBlocks, blockSize, dev_out, dev_blockSumsScanned);

            // post process to make it an exclusive scan and remove trailing zeros from non power of 2
            postScanFormat << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_in, dev_out);
            int* temp = dev_in;
            dev_in = dev_out;
            dev_out = temp;

            setFirstElementToZero << <1, 1 >> > (dataLength, dev_out);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeInBytes, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_blockSums);
            cudaFree(dev_blockSumsScanned);
        }
    }
}
