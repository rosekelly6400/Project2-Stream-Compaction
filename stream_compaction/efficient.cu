#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
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
                odata[idx] = idata[idx - zeroBufferSize];
            }

        }

        __global__ void postScanFormat(int nOriginal, int nPowerOf2, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            int zeroBufferSize = nPowerOf2 - nOriginal;
            if (idx + zeroBufferSize >= nPowerOf2) return;

            if (idx == 0)
            {
                odata[0] = 0;
            }
            else {
                odata[idx] = idata[idx + zeroBufferSize];
            }

        }

        __global__ void sweepUp(int n, int twoPowD, int spacing, int* odata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx = idx * spacing;
            if (idx + spacing - 1 >= n) return;

            odata[idx + spacing - 1] += odata[idx + twoPowD - 1];

        }

        __global__ void sweepDown(int n, int twoPowD, int spacing, int* odata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx = idx * spacing;
            if (idx >= n) return;

            int temp = odata[idx + twoPowD - 1];
            odata[idx + twoPowD - 1] = odata[idx + spacing - 1];
            odata[idx + spacing - 1] += temp;
        }

        __global__ void setLastElementToZero(int n, int* odata)
        {
            odata[n - 1] = 0;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int blockSize = 32;
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

            // set number of threads to be ceiling of log2(n)
            dim3 fullBlocksPerGrid((dataLength + blockSize - 1) / blockSize);


            // add zeros to beginning of buffer if n is not power of 2
            if (dataLength != n)
            {
                formatForNonPowOf2 << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_out, dev_in);
            }
            else
            {
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }
            // up-sweep
            for (int i = 0; i <= log2n-1; i++)
            {
                int spacing = 1 << ( i + 1);
                dim3 currIterBlocksPerGrid(((dataLength / spacing) + blockSize - 1) / blockSize);
                int twoPowD = 1 << i;
                sweepUp << <currIterBlocksPerGrid, blockSize >> > (dataLength, twoPowD, spacing, dev_out);
            }
            setLastElementToZero << <1, 1 >> > (dataLength, dev_out);
            // down-sweep
            for (int i = log2n - 1; i >= 0; i--)
            {
                int spacing = 1 << (i + 1);
                dim3 currIterBlocksPerGrid(((dataLength / spacing) + blockSize - 1) / blockSize);
                int twoPowD = 1 << i;
                sweepDown << <currIterBlocksPerGrid, blockSize >> > (dataLength, twoPowD, spacing, dev_out);
            }
            // post process to remove trailing zeros from non power of 2
            if (n != dataLength) {
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
                postScanFormat << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_out, dev_in);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeInBytes, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO
            int numValidElements;
            int blockSize = 32;
            // include zero buffer for non power of 2 n
            int log2n = ilog2ceil(n);
            int dataLength = 1 << log2n;
            int sizeInBytes = dataLength * sizeof(int);

            int* dev_in;
            int* dev_out;
            int* dev_boolMap;
            int* dev_indices;

            cudaMalloc((void**)&dev_in, sizeInBytes);
            checkCUDAError("cudaMalloc dev_in failed!");
            cudaMalloc((void**)&dev_out, sizeInBytes);
            checkCUDAError("cudaMalloc dev_out failed!");
            cudaMalloc((void**)&dev_boolMap, sizeInBytes);
            checkCUDAError("cudaMalloc dev_boolMap failed!");
            cudaMalloc((void**)&dev_indices, sizeInBytes);
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMemcpy(dev_in, idata, sizeInBytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // set number of threads to be ceiling of log2(n)
            dim3 fullBlocksPerGrid((dataLength + blockSize - 1) / blockSize);


            // add zeros to beginning of buffer if n is not power of 2
            if (dataLength != n)
            {
                formatForNonPowOf2 << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_out, dev_in);
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }
            // map to 0 and 1 to indicate valid value
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (dataLength, dev_boolMap, dev_in);
            // up-sweep
            StreamCompaction::Common::copyBuffer << <fullBlocksPerGrid, blockSize >> > (dataLength, dev_indices, dev_boolMap);
            for (int i = 0; i <= log2n - 1; i++)
            {
                int spacing = 1 << (i + 1);
                dim3 currIterBlocksPerGrid(((dataLength/ spacing)+blockSize - 1) / blockSize);
                int twoPowD = 1 << i;
                sweepUp << <currIterBlocksPerGrid, blockSize >> > (dataLength, twoPowD, spacing, dev_indices);
            }
            cudaMemcpy(&numValidElements, &(dev_indices[dataLength - 1]), sizeof(int), cudaMemcpyDeviceToHost);
            
            setLastElementToZero << <1, 1 >> > (dataLength, dev_indices);
            // down-sweep
            for (int i = log2n - 1; i >= 0; i--)
            {
                int spacing = 1 << (i + 1);
                dim3 currIterBlocksPerGrid(((dataLength / spacing) + blockSize - 1) / blockSize);
                int twoPowD = 1 << i;
                sweepDown << <currIterBlocksPerGrid, blockSize >> > (dataLength, twoPowD, spacing, dev_indices);
            }

            //SCATTER
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (dataLength, dev_out, dev_in, dev_boolMap, dev_indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeInBytes, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_boolMap);
            cudaFree(dev_indices);

            return numValidElements;
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

        __global__ void makeExclusive(int n, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if (idx == 0)
            {
                odata[idx] = 0;
            }
            else {
                odata[idx] = idata[idx - 1];
            }
        }

        __global__ void sweepUpSweepDown_sharedMem(int n, int blockSize, int log2BlockSize, int* odata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            unsigned sharedIdx = threadIdx.x;
            extern __shared__ int shared_idata[];
            __shared__ int temp[1024];
            shared_idata[sharedIdx] = odata[idx];
            __syncthreads();

            // sweep up
            for (int i = 0; i <= log2BlockSize - 1; i++)
            {
                int spacing = 1 << (i + 1);
                if (sharedIdx * spacing + spacing - 1 < blockSize)
                {
                    int twoPowD = 1 << i;
                    shared_idata[sharedIdx * spacing + spacing - 1] += shared_idata[sharedIdx * spacing + twoPowD - 1];
                }
                __syncthreads();
            }
            // store total for block
            int totalValueForBlock = -1;
            // set last element to 0
            if (sharedIdx == 0)
            {
                totalValueForBlock = shared_idata[blockSize - 1];
                shared_idata[blockSize - 1] = 0;
            }
            __syncthreads();
            for (int i = log2BlockSize - 1; i >= 0; i--)
            {
                int spacing = 1 << (i + 1);
                int twoPowD = 1 << i;
                if (sharedIdx * spacing + spacing - 1 < blockSize)
                {
                    int temp = shared_idata[sharedIdx * spacing + twoPowD - 1];
                    shared_idata[sharedIdx * spacing + twoPowD - 1] = shared_idata[sharedIdx * spacing + spacing - 1];
                    shared_idata[sharedIdx * spacing + spacing - 1] += temp;
                }

                __syncthreads();
            }

            // move everything back by one (maybe need another shared array)
            if (sharedIdx < n - 1)
            {
                temp[sharedIdx] = shared_idata[sharedIdx + 1];
            }
            __syncthreads();
            //set last value to totalValueForBlock
            if (sharedIdx == 0)
            {
                temp[blockSize - 1] = totalValueForBlock;
            }
            __syncthreads();
            odata[idx] = temp[sharedIdx];
        }

        // EXTRA CREDIT
        void scanShared(int n, int* odata, const int* idata) {
            // TODO
            int blockSize = 256;
            // include zero buffer for non power of 2 n
            int log2n = ilog2ceil(n);
            int dataLength = 1 << log2n;
            /*if (dataLength < blockSize) {
                blockSize = dataLength;
            }*/
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
            cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc dev_blockSums failed!");
            cudaMalloc((void**)&dev_blockSumsScanned, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc dev_blockSums failed!");

            cudaMemcpy(dev_in, idata, sizeInBytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // set number of threads to be ceiling of log2(n)
            dim3 fullBlocksPerGrid((dataLength + blockSize - 1) / blockSize);


            // add zeros to beginning of buffer if n is not power of 2
            if (dataLength != n)
            {
                formatForNonPowOf2 << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_out, dev_in);
            }
            else
            {
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }
            // 1. scan array as blocks of blockSize
            int sharedMemSizeInBytes = blockSize * sizeof(int);
            int log2BlockSize = ilog2ceil(blockSize);
            sweepUpSweepDown_sharedMem << <fullBlocksPerGrid, blockSize, sharedMemSizeInBytes >> > (dataLength, blockSize, log2BlockSize, dev_out);

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
            int* temp = dev_in;
            dev_in = dev_out;
            dev_out = temp;
            makeExclusive << <fullBlocksPerGrid, blockSize >> > (dataLength, dev_out, dev_in);

            // post process to make it an exclusive scan and remove trailing zeros from non power of 2
            if (n != dataLength) {
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
                postScanFormat << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_out, dev_in);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeInBytes, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_blockSums);
            cudaFree(dev_blockSumsScanned);
        }

        int compactShared(int n, int* odata, const int* idata) {
            // TODO
            int numValidElements;
            int blockSize = 256;
            // include zero buffer for non power of 2 n
            int log2n = ilog2ceil(n);
            int dataLength = 1 << log2n;
            int sizeInBytes = dataLength * sizeof(int);
            int numBlocks = (dataLength + blockSize - 1) / blockSize;

            int* dev_in;
            int* dev_out;
            int* dev_boolMap;
            int* dev_indices;
            int* dev_blockSums;
            int* dev_blockSumsScanned;

            cudaMalloc((void**)&dev_in, sizeInBytes);
            checkCUDAError("cudaMalloc dev_in failed!");
            cudaMalloc((void**)&dev_out, sizeInBytes);
            checkCUDAError("cudaMalloc dev_out failed!");
            cudaMalloc((void**)&dev_boolMap, sizeInBytes);
            checkCUDAError("cudaMalloc dev_boolMap failed!");
            cudaMalloc((void**)&dev_indices, sizeInBytes);
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_blockSums, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc dev_blockSums failed!");
            cudaMalloc((void**)&dev_blockSumsScanned, numBlocks * sizeof(int));
            checkCUDAError("cudaMalloc dev_blockSums failed!");
            cudaMemcpy(dev_in, idata, sizeInBytes, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // set number of threads to be ceiling of log2(n)
            dim3 fullBlocksPerGrid((dataLength + blockSize - 1) / blockSize);


            // add zeros to beginning of buffer if n is not power of 2
            if (dataLength != n)
            {
                formatForNonPowOf2 << <fullBlocksPerGrid, blockSize >> > (n, dataLength, dev_out, dev_in);
                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }
            // map to 0 and 1 to indicate valid value
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (dataLength, dev_boolMap, dev_in);
            StreamCompaction::Common::copyBuffer << <fullBlocksPerGrid, blockSize >> > (dataLength, dev_indices, dev_boolMap);

            // 1. scan array as blocks of blockSize
            int sharedMemSizeInBytes = blockSize * sizeof(int);
            int log2BlockSize = ilog2ceil(blockSize);
            sweepUpSweepDown_sharedMem << <fullBlocksPerGrid, blockSize, sharedMemSizeInBytes >> > (dataLength, blockSize, log2BlockSize, dev_indices);

            // 2. Write total sum of each block into a new array
            dim3 blockSumsBlocksPerGrid((numBlocks + blockSize - 1) / blockSize);
            StreamCompaction::Common::copyBlockSums << <blockSumsBlocksPerGrid, blockSize >> > (numBlocks, blockSize, dev_blockSums, dev_indices);
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
            StreamCompaction::Common::addBlockSumsBack << <fullBlocksPerGrid, blockSize >> > (dataLength, numBlocks, blockSize, dev_indices, dev_blockSumsScanned);
            cudaMemcpy(&numValidElements, &(dev_indices[dataLength - 1]), sizeof(int), cudaMemcpyDeviceToHost);
            int* temp = dev_out;
            dev_out = dev_indices;
            dev_indices = temp;
            makeExclusive << <fullBlocksPerGrid, blockSize >> > (dataLength, dev_indices, dev_out);



            //SCATTER
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (dataLength, dev_out, dev_in, dev_boolMap, dev_indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_out, sizeInBytes, cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_boolMap);
            cudaFree(dev_indices);
            cudaFree(dev_blockSums);
            cudaFree(dev_blockSumsScanned);

            return numValidElements;
        }
    }
}
