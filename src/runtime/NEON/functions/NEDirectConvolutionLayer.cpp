/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <cmath>
#include <tuple>

using namespace arm_compute;

NEDirectConvolutionLayer::NEDirectConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _output_stage_kernel(), _conv_kernel(), _input_border_handler(), _accumulator(), _has_bias(false), _is_fixed_point(false)
{
}

void NEDirectConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &conv_info)
{
    // Free accumulator
    if(_accumulator.buffer() != nullptr)
    {
        _accumulator.allocator()->free();
    }

    // Check if bias should be added in the convolution result
    _has_bias = (bias != nullptr);

    // Allocate the intermediate accumulator tensor in case of fixed point input
    _is_fixed_point = is_data_type_fixed_point(input->info()->data_type());
    if(_is_fixed_point)
    {
        const DataType promoted_dt = (input->info()->data_type() == DataType::QS8) ? DataType::QS16 : DataType::QS32;
        _accumulator.allocator()->init(TensorInfo(output->info()->tensor_shape(), 1, promoted_dt, output->info()->fixed_point_position()));
        _memory_group.manage(&_accumulator);
        _conv_kernel.configure(input, weights, &_accumulator, conv_info);

        // When no bias is provided, we need to downscale the accumulator tensor
        _output_stage_kernel.configure(&_accumulator, bias, output);
        _accumulator.allocator()->allocate();
    }
    else
    {
        _conv_kernel.configure(input, weights, output, conv_info);
        if(_has_bias)
        {
            _output_stage_kernel.configure(output, bias);
        }
    }

    // Add zero padding XY
    _input_border_handler.configure(input, _conv_kernel.border_size(), BorderMode::CONSTANT, PixelValue(static_cast<float>(0.f)));
}

Status NEDirectConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);

    DataType data_type = output->data_type();
    if(is_data_type_fixed_point(data_type))
    {
        // Promote data type in case of fixed point
        data_type = ((data_type == DataType::QS8) ? DataType::QS16 : DataType::QS32);
    }
    TensorInfo accumulator(output->clone()->set_is_resizable(true).reset_padding().set_data_type(data_type));

    // Validate Convolution kernel
    ARM_COMPUTE_RETURN_ON_ERROR(NEDirectConvolutionLayerKernel::validate(input, weights, &accumulator, conv_info));

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->dimension(0) != weights->dimension(3),
                                        "Biases size and number of input feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->num_dimensions() > 1, "Biases should be one dimensional");
    }

    // Validate bias kernel
    ARM_COMPUTE_RETURN_ON_ERROR(NEDirectConvolutionLayerOutputStageKernel::validate(&accumulator, bias, output));

    return Status{};
}

void NEDirectConvolutionLayer::run()
{
    NEScheduler::get().schedule(&_input_border_handler, Window::DimZ);

    _memory_group.acquire();

    NEScheduler::get().schedule(&_conv_kernel, Window::DimZ);
    if(_has_bias || _is_fixed_point)
    {
        NEScheduler::get().schedule(&_output_stage_kernel, Window::DimY);
    }
    _memory_group.release();
}
