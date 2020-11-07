/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLMinMaxLocation.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "src/core/CL/kernels/CLMinMaxLocationKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
CLMinMaxLocation::CLMinMaxLocation()
    : _min_max_kernel(support::cpp14::make_unique<CLMinMaxKernel>()),
      _min_max_loc_kernel(support::cpp14::make_unique<CLMinMaxLocationKernel>()),
      _min_max_vals(),
      _min_max_count_vals(),
      _min(nullptr),
      _max(nullptr),
      _min_count(nullptr),
      _max_count(nullptr),
      _min_loc(nullptr),
      _max_loc(nullptr)
{
}

CLMinMaxLocation::~CLMinMaxLocation() = default;

void CLMinMaxLocation::configure(const ICLImage *input, void *min, void *max, CLCoordinates2DArray *min_loc, CLCoordinates2DArray *max_loc, uint32_t *min_count, uint32_t *max_count)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, min, max, min_loc, max_loc, min_count, max_count);
}

void CLMinMaxLocation::configure(const CLCompileContext &compile_context, const ICLImage *input, void *min, void *max, CLCoordinates2DArray *min_loc, CLCoordinates2DArray *max_loc,
                                 uint32_t *min_count,
                                 uint32_t *max_count)
{
    ARM_COMPUTE_ERROR_ON(nullptr == min);
    ARM_COMPUTE_ERROR_ON(nullptr == max);

    _min_max_vals       = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, 2 * sizeof(int32_t));
    _min_max_count_vals = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, 2 * sizeof(uint32_t));
    _min                = min;
    _max                = max;
    _min_count          = min_count;
    _max_count          = max_count;
    _min_loc            = min_loc;
    _max_loc            = max_loc;

    _min_max_kernel->configure(compile_context, input, &_min_max_vals);
    _min_max_loc_kernel->configure(compile_context, input, &_min_max_vals, &_min_max_count_vals, _min_loc, _max_loc);
}

void CLMinMaxLocation::run()
{
    cl::CommandQueue q = CLScheduler::get().queue();

    CLScheduler::get().enqueue(*_min_max_kernel, false);
    CLScheduler::get().enqueue(*_min_max_loc_kernel, false);

    // Update min and max
    q.enqueueReadBuffer(_min_max_vals, CL_FALSE, 0 * sizeof(int32_t), sizeof(int32_t), static_cast<int32_t *>(_min));
    q.enqueueReadBuffer(_min_max_vals, CL_FALSE, 1 * sizeof(int32_t), sizeof(int32_t), static_cast<int32_t *>(_max));

    // Update min and max count
    if(_min_count != nullptr)
    {
        q.enqueueReadBuffer(_min_max_count_vals, CL_FALSE, 0 * sizeof(uint32_t), sizeof(uint32_t), _min_count);
    }
    if(_max_count != nullptr)
    {
        q.enqueueReadBuffer(_min_max_count_vals, CL_FALSE, 1 * sizeof(uint32_t), sizeof(uint32_t), _max_count);
    }

    // Update min/max point arrays (Makes the kernel blocking)
    if(_min_loc != nullptr)
    {
        unsigned int min_count = 0;
        q.enqueueReadBuffer(_min_max_count_vals, CL_TRUE, 0 * sizeof(uint32_t), sizeof(uint32_t), &min_count);
        size_t min_corner_size = std::min(static_cast<size_t>(min_count), _min_loc->max_num_values());
        _min_loc->resize(min_corner_size);
    }
    if(_max_loc != nullptr)
    {
        unsigned int max_count = 0;
        q.enqueueReadBuffer(_min_max_count_vals, CL_TRUE, 1 * sizeof(uint32_t), sizeof(uint32_t), &max_count);
        size_t max_corner_size = std::min(static_cast<size_t>(max_count), _max_loc->max_num_values());
        _max_loc->resize(max_corner_size);
    }
}
} // namespace arm_compute
