#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            //thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

            thrust::host_vector<int> in(idata, idata + n);
            thrust::host_vector<int> out(odata, odata + n);

            thrust::device_vector<int> dev_in(in.begin(), in.begin() + n);
            thrust::device_vector<int> dev_out = out;

            timer().startGpuTimer();

            thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());

            timer().endGpuTimer();

            thrust::copy(dev_out.begin(), dev_out.end(), odata);

        }

        struct is_zero
        {
            __host__ __device__
                bool operator()(const int x)
            {
                return x == 0;
            }
        };

        int compact(int n, int* odata, const int* idata) {

            thrust::device_vector<int> dev_in(idata, idata + n);
            thrust::device_vector<int> dev_out(odata, odata + n);

            timer().startGpuTimer();

            auto new_end = thrust::remove_if(dev_in.begin(), dev_in.end(), is_zero());

            timer().endGpuTimer();

            thrust::copy(dev_in.begin(), new_end, odata);

            int length = new_end - dev_in.begin();

            return length;
        }
    }
}
