#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        void scanHelper_notimer(int n, int* odata, const int* idata) {
            // TODO
            odata[0] = 0;
            for (int k = 1; k < n; ++k)
            {
                odata[k] = odata[k - 1] + idata[k - 1];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            scanHelper_notimer(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int numValidElements = 0;
            for (int k = 0; k < n; ++k)
            {
                if (idata[k] != 0)
                {
                    odata[numValidElements] = idata[k];
                    numValidElements++;
                }
            }
            timer().endCpuTimer();
            return numValidElements;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            //map input array to 0s and 1s
            int* temp = new int[n];
            for (int k = 0; k < n; ++k)
            {
                if (idata[k] != 0)
                {
                    temp[k] = 1;
                }
                else {
                    temp[k] = 0;
                }
            }
            //scan
            scanHelper_notimer(n, odata, temp);
            int numValidElements = odata[n-1];
            if (idata[n - 1] != 0) {
                numValidElements++;
            }
            //scatter
            for (int k = 0; k < n; ++k)
            {
                if (temp[k] == 1)
                {
                    odata[odata[k]] = idata[k];
                }
            }
            delete[] temp;
            timer().endCpuTimer();
            return numValidElements;
        }
    }
}
