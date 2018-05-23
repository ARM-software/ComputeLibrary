/*
 * Copyright (c) 2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_TEST_OPENCL_MEMORY_USAGE
#define ARM_COMPUTE_TEST_OPENCL_MEMORY_USAGE

#include "Instrument.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/core/CL/OpenCL.h"
#endif /* ARM_COMPUTE_CL */

#include <list>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Instrument collecting memory usage information for OpenCL*/
class OpenCLMemoryUsage : public Instrument
{
public:
    /** Construct an OpenCL timer.
     *
     * @param[in] scale_factor Measurement scale factor.
     */
    OpenCLMemoryUsage(ScaleFactor scale_factor);
    std::string     id() const override;
    void            test_start() override;
    void            start() override;
    void            stop() override;
    void            test_stop() override;
    MeasurementsMap test_measurements() const override;
    MeasurementsMap measurements() const override;
#ifdef ARM_COMPUTE_CL
    std::function<decltype(clCreateBuffer)>     real_clCreateBuffer;
    std::function<decltype(clRetainMemObject)>  real_clRetainMemObject;
    std::function<decltype(clReleaseMemObject)> real_clReleaseMemObject;
    std::function<decltype(clSVMAlloc)>         real_clSVMAlloc;
    std::function<decltype(clSVMFree)>          real_clSVMFree;

private:
    float _scale_factor{};
    struct Allocation
    {
        Allocation() = default;
        Allocation(size_t alloc_size)
            : size(alloc_size)
        {
        }
        size_t size{ 0 };
        int    refcount{ 1 };
    };
    std::map<cl_mem, Allocation> _allocations;
    std::map<void *, size_t>     _svm_allocations;
    struct Stats
    {
        size_t total_allocated{ 0 };
        size_t max_in_use{ 0 };
        size_t in_use{ 0 };
        size_t num_allocations{ 0 };
    } _start, _end, _now;
#endif /* ARM_COMPUTE_CL */
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_OPENCL_MEMORY_USAGE */
