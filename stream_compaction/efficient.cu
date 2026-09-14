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
            if (idx >= nPowerOf2) return;
            int zeroBufferSize = nPowerOf2 - nOriginal;

            if (idx == 0)
            {
                odata[0] = 0;
            }
            else {
                odata[idx] = idata[idx + zeroBufferSize];
            }

        }

        __global__ void sweepUp(int n, int iteration, int spacing, int log2n, int* odata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx = idx * spacing;
            if (idx >= n) return;

            int twoPowD = std::pow(2, iteration);
            odata[idx + spacing - 1] += odata[idx + twoPowD - 1];

        }

        __global__ void sweepUp_indata(int n, int iteration, int log2n, int* odata, int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            int spacing = std::pow(2, iteration + 1);
            idx = idx * spacing;
            if (idx >= n) return;

            int twoPowD = std::pow(2, iteration);
            odata[idx + spacing - 1] = idata[idx + spacing - 1] + idata[idx + twoPowD - 1];

        }

        __global__ void sweepDown(int n, int iteration, int spacing, int log2n, int* odata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx = idx * spacing;
            if (idx >= n) return;

            int twoPowD = std::pow(2, iteration);
            int temp = odata[idx + twoPowD - 1];
            if (iteration == log2n -1 && idx + twoPowD - 1 == n - 1) {
                temp = 0;
            }
            odata[idx + twoPowD - 1] = odata[idx + spacing - 1];
            if (iteration == log2n - 1 && idx + spacing - 1 == n - 1) {
                odata[idx + twoPowD - 1] = 0;
            }
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
                int spacing = std::pow(2, i + 1);
                dim3 currIterBlocksPerGrid(((dataLength / spacing) + blockSize - 1) / blockSize);
                sweepUp << <currIterBlocksPerGrid, blockSize >> > (dataLength, i, spacing, log2n, dev_out);
            }
            setLastElementToZero << <1, 1 >> > (dataLength, dev_out);
            // down-sweep
            for (int i = log2n - 1; i >= 0; i--)
            {
                int spacing = std::pow(2, i + 1);
                dim3 currIterBlocksPerGrid(((dataLength / spacing) + blockSize - 1) / blockSize);
                sweepDown << <currIterBlocksPerGrid, blockSize >> > (dataLength, i, spacing, log2n, dev_out);
            }
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
            int dataLength = std::pow(2, log2n);
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
                int spacing = std::pow(2, i + 1);
                dim3 currIterBlocksPerGrid(((dataLength/ spacing)+blockSize - 1) / blockSize);
                sweepUp << <currIterBlocksPerGrid, blockSize >> > (dataLength, i, spacing, log2n, dev_indices);
            }
            cudaMemcpy(&numValidElements, &(dev_indices[dataLength - 1]), sizeof(int), cudaMemcpyDeviceToHost);
            
            setLastElementToZero << <1, 1 >> > (dataLength, dev_indices);
            // down-sweep
            for (int i = log2n - 1; i >= 0; i--)
            {
                int spacing = std::pow(2, i + 1);
                dim3 currIterBlocksPerGrid(((dataLength / spacing) + blockSize - 1) / blockSize);
                sweepDown << <currIterBlocksPerGrid, blockSize >> > (dataLength, i, spacing, log2n, dev_indices);
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
    }
}
