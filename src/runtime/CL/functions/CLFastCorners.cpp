/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLFastCorners.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/CL/kernels/CLFastCornersKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/ITensorAllocator.h"

#include <algorithm>
#include <cstring>

using namespace arm_compute;

CLFastCorners::CLFastCorners(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _fast_corners_kernel(),
      _suppr_func(),
      _copy_array_kernel(),
      _output(),
      _suppr(),
      _win(),
      _non_max(false),
      _num_corners(nullptr),
      _num_buffer(),
      _corners(nullptr),
      _constant_border_value(0)
{
}

void CLFastCorners::configure(const ICLImage *input, float threshold, bool nonmax_suppression, ICLKeyPointArray *corners,
                              unsigned int *num_corners, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(BorderMode::UNDEFINED != border_mode);
    ARM_COMPUTE_ERROR_ON(nullptr == corners);
    ARM_COMPUTE_ERROR_ON(threshold < 1 && threshold > 255);

    TensorInfo tensor_info(input->info()->tensor_shape(), 1, DataType::U8);
    _output.allocator()->init(tensor_info);

    _non_max               = nonmax_suppression;
    _num_corners           = num_corners;
    _corners               = corners;
    _num_buffer            = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(unsigned int));
    _constant_border_value = constant_border_value;

    const bool update_number = (nullptr != _num_corners);

    _memory_group.manage(&_output);
    _fast_corners_kernel.configure(input, &_output, threshold, nonmax_suppression, border_mode);

    if(!_non_max)
    {
        _copy_array_kernel.configure(&_output, update_number, _corners, &_num_buffer);
    }
    else
    {
        _suppr.allocator()->init(tensor_info);
        _memory_group.manage(&_suppr);

        _suppr_func.configure(&_output, &_suppr, border_mode);
        _copy_array_kernel.configure(&_suppr, update_number, _corners, &_num_buffer);

        _suppr.allocator()->allocate();
    }

    // Allocate intermediate tensors
    _output.allocator()->allocate();
}

void CLFastCorners::run()
{
    cl::CommandQueue q = CLScheduler::get().queue();

    MemoryGroupResourceScope scope_mg(_memory_group);

    if(_non_max)
    {
        ARM_COMPUTE_ERROR_ON_MSG(_output.cl_buffer().get() == nullptr, "Unconfigured function");
        const auto out_buffer = static_cast<unsigned char *>(q.enqueueMapBuffer(_output.cl_buffer(), CL_TRUE, CL_MAP_WRITE, 0, _output.info()->total_size()));
        memset(out_buffer, 0, _output.info()->total_size());
        q.enqueueUnmapMemObject(_output.cl_buffer(), out_buffer);
    }

    CLScheduler::get().enqueue(_fast_corners_kernel, false);

    if(_non_max)
    {
        _suppr_func.run();
    }

    CLScheduler::get().enqueue(_copy_array_kernel, false);

    unsigned int get_num_corners = 0;
    q.enqueueReadBuffer(_num_buffer, CL_TRUE, 0, sizeof(unsigned int), &get_num_corners);

    size_t corner_size = std::min(static_cast<size_t>(get_num_corners), _corners->max_num_values());

    _corners->resize(corner_size);

    if(_num_corners != nullptr)
    {
        *_num_corners = get_num_corners;
    }

    q.flush();
}
