CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Rose Kelly
  * [LinkedIn](www.linkedin.com/in/rose-kelly-b480b01a8)
* Tested on: Windows 11, i7-13700F @ 2.10 GHz 16GB, RTX 4060-Ti 8GB (Personal Computer)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)


1. Update all of the TODOs at the top of your `README.md`.
2. Add a description of this project including a list of its features.
3. Add your performance analysis (see below).

All extra credit features must be documented in your `README.md`, explaining its
value (with performance comparison, if applicable!) and showing an example how
it works. For radix sort, show how it is called and an example of its output.

Always profile with Release mode builds and run without debugging.

### Questions

* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.
  * (You shouldn't compare unoptimized implementations to each other!)
  I optimized my block sizes for each implementation by looking at the nsight compute report when just that implementation was running and checking the times given in the program output. I ran the program with `2^25` elements when comparing times as this fairly large number gave me more consistent timing data. 
  
  For the naive implementation, I saw the best performance with a block size of 1024. This had the highest Memory Throughput and Compute Throughput in Nsight Compute compared to other sizes. This makes sense as the larger block size may allow the scheduler to hide latency from the many global memory reads with other warps.

  For the global memory work efficient approach, I saw the best performance with a block size of 32. This had the highest Memory Throughput in Nsight Compute (by just 0.15%) compared to other sizes and tied with 1024 for Compute Throughput. This may be because during the sweep up and sweep down, the number of actual threads required varies and gets to be below 1024 and even below 32. While my kernels exit early for global indices (combined block and thread index) outside the buffer size, there are still calculations in the kernel required to determine that. These few extra calculations on the unneeded threads being created at greater block sizes could be responsible for dragging performance slightly for larger block sizes, and also explain why block size of 32 has a slightly higher compute throughput.

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * We wrapped up both CPU and GPU timing functions as a performance timer class for you to conveniently measure the time cost.
    * We use `std::chrono` to provide CPU high-precision timing and CUDA event to measure the CUDA performance.
    * For CPU, put your CPU code between `timer().startCpuTimer()` and `timer().endCpuTimer()`.
    * For GPU, put your CUDA code between `timer().startGpuTimer()` and `timer().endGpuTimer()`. Be sure **not** to include any *initial/final* memory operations (`cudaMalloc`, `cudaMemcpy`) in your performance measurements, for comparability.
    * Don't mix up `CpuTimer` and `GpuTimer`.
  * To guess at what might be happening inside the Thrust implementation (e.g.
    allocation, memory copy), take a look at the Nsight timeline for its
    execution. Your analysis here doesn't have to be detailed, since you aren't
    even looking at the code for the implementation.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

For the naive implementation, Compute Throughput was fairly low (~10%) while memory throughput was high (~94%). This indicates memory I/O is the bottleneck since the compute throughput is likely lower due to wiating around for memory reads. This fits with the naive implementation since it relies heavily on global memory reads.

For the work efficient implementation, the Compute Throughput was even lower (~5%) while memory throughput was only slightly lower (~92%). This similarly indicates a memory I/O bottleneck. The global memory work efficient approach also makes many many global memory reads. My work efficient compact also has additional device buffers other than the the in and out buffers, as well as kernels to copy and scatter between buffers which requires more global memory reads. Additionally, the work efficient implementation should have fewer computations than the naive approach, so proportionally less time may be spent on compute rather than memory fetching.

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

These questions should help guide you in performance analysis on future
assignments, as well.
