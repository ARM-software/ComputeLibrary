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
#include "arm_compute/core/NEON/kernels/NEDepthwiseVectorToTensorKernel.h"

#include "arm_compute/core/CPP/Validate.h"
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, size_t conv_w, size_t conv_h)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::S32, DataType::F16, DataType::F32);

    if(output->total_size() != 0)
    {
        TensorShape output_shape = compute_vector_to_tensor_output_shape(input->tensor_shape(), conv_w, conv_h, output->data_layout());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

template <typename T>
void NEDepthwiseVectorToTensorKernel::vector_to_tensor(const Window &window)
{
    // const int input_w         = _input->info()->dimension(0);
    const int output_stride_x = _output->info()->strides_in_bytes().x();
    const int output_stride_y = _output->info()->strides_in_bytes().y();
    const int output_stride_z = _output->info()->strides_in_bytes().z();

    // Setup output window
    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Iterator in(_input, window);
    Iterator out(_output, window_out);

    const int patch_size = _conv_dims.first * _conv_dims.second;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int z       = id.x() / patch_size;
        const int index2D = id.x() - z * patch_size;

        auto input_ptr  = reinterpret_cast<T *>(in.ptr());
        auto output_ptr = reinterpret_cast<T *>(out.ptr() + index2D % _conv_dims.first * output_stride_x + index2D / _conv_dims.first * output_stride_y + z * output_stride_z);

        *output_ptr = *input_ptr;
    },
    in, out);
}

NEDepthwiseVectorToTensorKernel::NEDepthwiseVectorToTensorKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _conv_dims()
{
}

void NEDepthwiseVectorToTensorKernel::configure(const ITensor *input, ITensor *output, size_t conv_w, size_t conv_h)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto inizialitation if not yet initialized
    TensorShape output_shape = compute_vector_to_tensor_output_shape(input->info()->tensor_shape(), conv_w, conv_h, output->info()->data_layout());
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), conv_w, conv_h));

    _input     = input;
    _output    = output;
    _conv_dims = std::pair<size_t, size_t>(conv_w, conv_h);

    // Set appropriate function to run
    switch(input->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = &NEDepthwiseVectorToTensorKernel::vector_to_tensor<uint8_t>;
            break;
        case DataType::S32:
            _func = &NEDepthwiseVectorToTensorKernel::vector_to_tensor<int32_t>;
            break;
        case DataType::F16:
            _func = &NEDepthwiseVectorToTensorKernel::vector_to_tensor<half>;
            break;
        case DataType::F32:
            _func = &NEDepthwiseVectorToTensorKernel::vector_to_tensor<float>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    // The NEDepthwisevectorToTensorKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

Status NEDepthwiseVectorToTensorKernel::validate(const ITensorInfo *input, const ITensorInfo *output, size_t conv_w, size_t conv_h)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, conv_w, conv_h));
    return Status{};
}

void NEDepthwiseVectorToTensorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_func != nullptr)
    {
        (this->*_func)(window);
    }
}
