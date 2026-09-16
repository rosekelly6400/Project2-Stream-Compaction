#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            // TODO
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            if (idata[idx] != 0)
            {
                bools[idx] = 1;
            }
            else {
                bools[idx] = 0;
            }
        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata, const int *idata, const int *bools, const int *indices) {
            // TODO
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            if (bools[idx] == 1)
            {
                odata[indices[idx]] = idata[idx];
            }
        }

        __global__ void copyBuffer(int n, int* odata, int* idata) {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            odata[idx] = idata[idx];
        }


        __global__ void copyBlockSums(int numBlocks, int blockSize, int* odata, const int* idata)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned originalBufferIdx = ((idx + 1) * blockSize) - 1;
            if (idx >= numBlocks) return;

            odata[idx] = idata[originalBufferIdx];
        }

        __global__ void addBlockSumsBack(int n, int numBlocks, int blockSize, int* odata, const int* blockSums)
        {
            unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            // subtract 1 for inclusive scan since nothing gets added to first block
            unsigned blockSumIdx = (idx / blockSize) - 1;
            if (idx >= n || blockSumIdx < 0 || blockSumIdx >= numBlocks) return;

            odata[idx] += blockSums[blockSumIdx];
        }
    }
}
