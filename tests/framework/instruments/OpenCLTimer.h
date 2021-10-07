/*
 * Copyright (c) 2017-2018, 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_OPENCL_TIMER
#define ARM_COMPUTE_TEST_OPENCL_TIMER

#include "Instrument.h"

#ifdef ARM_COMPUTE_CL
#include "arm_compute/core/CL/OpenCL.h"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/graph/Workload.h"

#include <list>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Instrument creating measurements based on the information returned by clGetEventProfilingInfo for each OpenCL kernel executed*/
template <bool output_timestamps>
class OpenCLClock : public Instrument
{
public:
    /** Construct an OpenCL timer.
     *
     * @param[in] scale_factor Measurement scale factor.
     */
    OpenCLClock(ScaleFactor scale_factor);
    std::string     id() const override;
    void            test_start() override;
    void            start() override;
    void            stop() override;
    void            test_stop() override;
    MeasurementsMap measurements() const override;
    MeasurementsMap test_measurements() const override;

private:
#ifdef ARM_COMPUTE_CL
    struct kernel_info
    {
        ::cl::Event event{}; /**< OpenCL event associated to the kernel enqueue */
        std::string name{};  /**< OpenCL Kernel name */
    };
    std::list<kernel_info>                          _kernels;
    std::function<decltype(clEnqueueNDRangeKernel)> _real_function;
#ifdef ARM_COMPUTE_GRAPH_ENABLED
    std::function<decltype(graph::execute_task)> _real_graph_function;
#endif /* ARM_COMPUTE_GRAPH_ENABLED */
    std::string _prefix;
    bool        _timer_enabled;
#endif /* ARM_COMPUTE_CL */

private:
    float _scale_factor{};
};

using OpenCLTimer      = OpenCLClock<false>;
using OpenCLTimestamps = OpenCLClock<true>;

} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_OPENCL_TIMER */
