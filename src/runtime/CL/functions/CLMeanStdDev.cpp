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
#include "arm_compute/core/TensorInfo.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLMeanStdDev.h"

using namespace arm_compute;

CLMeanStdDev::CLMeanStdDev(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _data_type(),
      _num_pixels(),
      _run_stddev(),
      _reduction_operation_mean(),
      _reduction_operation_stddev(),
      _reduction_output_mean(),
      _reduction_output_stddev(),
      _mean(nullptr),
      _stddev(nullptr),
      _mean_stddev_kernel(),
      _fill_border_kernel(),
      _global_sum(),
      _global_sum_squared()
{
}

Status CLMeanStdDev::validate(ITensorInfo *input, float *mean, float *stddev)
{
    ARM_COMPUTE_RETURN_ERROR_ON_TENSOR_NOT_2D(input);
    if(is_data_type_float(input->data_type()))
    {
        ARM_COMPUTE_UNUSED(mean);
        ARM_COMPUTE_UNUSED(stddev);

        TensorShape output_shape      = TensorShape{ 1, input->dimension(1) };
        TensorInfo  output_shape_info = TensorInfo(output_shape, 1, DataType::U8);
        return CLReductionOperation::validate(input, &output_shape_info, 0, ReductionOperation::SUM);
    }
    else
    {
        return CLMeanStdDevKernel::validate(input, mean, nullptr, stddev, nullptr);
    }
}

void CLMeanStdDev::configure(ICLImage *input, float *mean, float *stddev)
{
    // In the case of F16/F32 we call reduction operation for calculating CLMeanStdDev
    _data_type = input->info()->data_type();

    if(is_data_type_float(_data_type))
    {
        _num_pixels = input->info()->dimension(0) * input->info()->dimension(1);

        _memory_group.manage(&_reduction_output_mean);
        _reduction_operation_mean.configure(input, &_reduction_output_mean, 0, ReductionOperation::SUM);
        _reduction_output_mean.allocator()->allocate();
        _mean = mean;

        if(stddev != nullptr)
        {
            _memory_group.manage(&_reduction_output_stddev);
            _reduction_operation_stddev.configure(input, &_reduction_output_stddev, 0, ReductionOperation::SUM_SQUARE);
            _reduction_output_stddev.allocator()->allocate();
            _stddev     = stddev;
            _run_stddev = true;
        }
    }
    else
    {
        _global_sum = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_ulong));

        if(stddev != nullptr)
        {
            _global_sum_squared = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_ulong));
        }

        _mean_stddev_kernel.configure(input, mean, &_global_sum, stddev, &_global_sum_squared);
        _fill_border_kernel.configure(input, _mean_stddev_kernel.border_size(), BorderMode::CONSTANT, PixelValue(static_cast<uint8_t>(0)));
    }
}

template <typename T>
void CLMeanStdDev::run_float()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    // Perform reduction on x-axis
    _reduction_operation_mean.run();
    if(_run_stddev)
    {
        _reduction_operation_stddev.run();
        _reduction_output_stddev.map(true);
    }

    _reduction_output_mean.map(true);

    auto mean = static_cast<T>(0);

    // Calculate final result for mean
    for(unsigned int i = 0; i < _reduction_output_mean.info()->dimension(1); ++i)
    {
        mean += *reinterpret_cast<T *>(_reduction_output_mean.buffer() + _reduction_output_mean.info()->offset_element_in_bytes(Coordinates(0, i)));
    }

    mean /= _num_pixels;
    *_mean = mean;

    if(_run_stddev)
    {
        auto stddev = static_cast<T>(0);
        // Calculate final result for stddev
        for(unsigned int i = 0; i < _reduction_output_stddev.info()->dimension(1); ++i)
        {
            stddev += *reinterpret_cast<T *>(_reduction_output_stddev.buffer() + _reduction_output_stddev.info()->offset_element_in_bytes(Coordinates(0, i)));
        }
        *_stddev = std::sqrt((stddev / _num_pixels) - (mean * mean));

        _reduction_output_stddev.unmap();
    }
    _reduction_output_mean.unmap();
}

void CLMeanStdDev::run_int()
{
    CLScheduler::get().enqueue(_fill_border_kernel);
    CLScheduler::get().enqueue(_mean_stddev_kernel);
}

void CLMeanStdDev::run()
{
    switch(_data_type)
    {
        case DataType::F16:
            run_float<half>();
            break;
        case DataType::F32:
            run_float<float>();
            break;
        case DataType::U8:
            run_int();
            break;
        default:
            ARM_COMPUTE_ERROR_ON("Not supported");
    }
}
