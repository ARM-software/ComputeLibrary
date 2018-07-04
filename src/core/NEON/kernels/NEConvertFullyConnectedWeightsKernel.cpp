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
#include "arm_compute/core/NEON/kernels/NEConvertFullyConnectedWeightsKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"

using namespace arm_compute;

NEConvertFullyConnectedWeightsKernel::NEConvertFullyConnectedWeightsKernel()
    : _input(nullptr), _output(nullptr), _factor1(0), _factor2(0)
{
}

void NEConvertFullyConnectedWeightsKernel::configure(const ITensor *input, ITensor *output, const TensorShape &original_input_shape,
                                                     DataLayout data_layout)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEConvertFullyConnectedWeightsKernel::validate(input->info(), output->info(), original_input_shape, data_layout));

    _input  = input;
    _output = output;

    const unsigned int num_elems_per_input_plane = original_input_shape.x() * original_input_shape.y();
    const unsigned int num_channels              = original_input_shape.z();

    // Set build options
    if(data_layout == DataLayout::NCHW)
    {
        _factor1 = num_elems_per_input_plane;
        _factor2 = num_channels;
    }
    else
    {
        _factor1 = num_channels;
        _factor2 = num_elems_per_input_plane;
    }

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    INEKernel::configure(win);
}

Status NEConvertFullyConnectedWeightsKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const TensorShape &original_input_shape,
                                                      DataLayout data_layout)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8, DataType::U16, DataType::S16, DataType::U32, DataType::S32,
                                                         DataType::QS32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(1) != original_input_shape.total_size_lower(3));
    ARM_COMPUTE_RETURN_ERROR_ON(data_layout == DataLayout::UNKNOWN);

    return Status{};
}

template <typename T>
void NEConvertFullyConnectedWeightsKernel::run_convert_fc_weights(const Window &window)
{
    const unsigned int dst_stride_x = _output->info()->strides_in_bytes().x();
    const unsigned int dst_stride_y = _output->info()->strides_in_bytes().y();

    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        *reinterpret_cast<T *>(output.ptr() + id.x() * dst_stride_x + (id.y() % _factor1 * _factor2 + id.y() / _factor1) * dst_stride_y) = *reinterpret_cast<T *>(input.ptr());
    },
    input);
}

void NEConvertFullyConnectedWeightsKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    switch(_input->info()->element_size())
    {
        case 1:
            run_convert_fc_weights<uint8_t>(window);
            break;
        case 2:
            run_convert_fc_weights<uint16_t>(window);
            break;
        case 4:
            run_convert_fc_weights<uint32_t>(window);
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not supported.");
            break;
    }
}
