/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDepthwiseWeightsReshapeKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

namespace
{
template <typename T>
void weights_reshape(const ITensor *input, const ITensor *bias, ITensor *output, const Window &window)
{
    const int input_w         = input->info()->dimension(0);
    const int output_stride_x = output->info()->strides_in_bytes().x();
    const int output_stride_y = output->info()->strides_in_bytes().y();

    Window window_in(window);
    // The first three dimensions of the input are increased by the inner loops
    window_in.set(Window::DimX, Window::Dimension(0, input->info()->dimension(0), input->info()->dimension(0)));
    window_in.set(Window::DimY, Window::Dimension(0, input->info()->dimension(1), 1));
    window_in.set(Window::DimZ, Window::Dimension(0, input->info()->dimension(2), 1));

    // Setup output window
    Window window_out;
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(input, window_in);
    Iterator out(output, window_out);

    execute_window_loop(window_in, [&](const Coordinates & id)
    {
        auto input_ptr  = reinterpret_cast<T *>(in.ptr());
        auto output_ptr = reinterpret_cast<T *>(out.ptr() + id.y() * input_w * output_stride_x + id.z() * output_stride_y);

        for(int i = 0; i < input_w; ++i, ++input_ptr)
        {
            *(output_ptr + i) = *input_ptr;
        }

        if(bias != nullptr)
        {
            *(output_ptr + input_w) = *(reinterpret_cast<T *>(bias->ptr_to_element(Coordinates(id.z()))));
        }
    },
    in, out);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *biases)
{
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(input->data_type()) && (biases != nullptr));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(2) != output->dimension(1));
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(0) != (input->dimension(0) * input->dimension(1) + ((biases != nullptr) ? 1 : 0)));

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != input->dimension(2));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

NEDepthwiseWeightsReshapeKernel::NEDepthwiseWeightsReshapeKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _biases(nullptr)
{
}

void NEDepthwiseWeightsReshapeKernel::configure(const ITensor *input, ITensor *output, const ITensor *biases)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), (biases != nullptr) ? biases->info() : nullptr));

    _input  = input;
    _output = output;
    _biases = biases;

    switch(_input->info()->element_size())
    {
        case 4:
        {
            _func = &weights_reshape<uint32_t>;
            break;
        }
        case 2:
        {
            _func = &weights_reshape<uint16_t>;
            break;
        }
        case 1:
        {
            _func = &weights_reshape<uint8_t>;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR_ON("Element size not supported");
            break;
        }
    }

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    // The NEDepthwiseWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

Status NEDepthwiseWeightsReshapeKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *biases)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, biases));
    return Status{};
}

void NEDepthwiseWeightsReshapeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_func != nullptr)
    {
        (*_func)(_input, _biases, _output, window);
    }
}
