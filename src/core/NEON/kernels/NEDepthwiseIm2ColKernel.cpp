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
#include "arm_compute/core/NEON/kernels/NEDepthwiseIm2ColKernel.h"

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
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, unsigned int depth_multiplier, const Size2D &dilation)
{
    ARM_COMPUTE_UNUSED(conv_info);
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(input->data_type()) && has_bias);
    ARM_COMPUTE_RETURN_ERROR_ON((input->dimension(2) * depth_multiplier) != output->dimension(2));
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(0) != (kernel_dims.width * kernel_dims.height + ((has_bias) ? 1 : 0)));
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || dilation.y() < 1);

    return Status{};
}
} // namespace

template <typename T>
void NEDepthwiseIm2ColKernel::run_generic(const Window &window)
{
    const int input_w        = _input->info()->dimension(0);
    const int input_h        = _input->info()->dimension(1);
    const int input_stride_x = _input->info()->strides_in_bytes().x();
    const int input_stride_y = _input->info()->strides_in_bytes().y();
    const int input_stride_z = _input->info()->strides_in_bytes().z();
    const int stride_x       = _conv_info.stride().first;
    const int stride_y       = _conv_info.stride().second;

    const int pad_left  = _conv_info.pad_left();
    const int pad_right = _conv_info.pad_right();
    const int pad_top   = _conv_info.pad_top();

    Window window_in(window);
    // The first three dimensions of the input are increased by the inner loops
    window_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Setup output window
    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, _output->info()->dimension(0), _output->info()->dimension(0)));
    window_out.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1), 1));
    window_out.set(Window::DimZ, Window::Dimension(0, _output->info()->dimension(2), 1));

    Iterator in(_input, window_in);
    Iterator out(_output, window_out);

    const int full_length   = input_w + pad_left + pad_right;
    const int max_initial_x = stride_x * (((full_length - (_kernel_dims.width + (_kernel_dims.width - 1) * (_dilation.x() - 1))) / stride_x) + 1);

    // Define pad value
    auto zero = static_cast<T>(0);
    if(std::is_same<T, uint8_t>::value)
    {
        zero = _input->info()->quantization_info().offset;
    }

    execute_window_loop(window_out, [&](const Coordinates & id)
    {
        const int src_pixel_linear = id.y() * stride_x;

        const int src_x = -pad_left + src_pixel_linear % max_initial_x;
        const int src_y = -pad_top + src_pixel_linear / max_initial_x * stride_y;

        // Get pointers
        const uint8_t *const input_ptr  = in.ptr() + id.z() / _depth_multiplier * input_stride_z;
        auto                 output_ptr = reinterpret_cast<T *>(out.ptr());
        const int            height     = src_y + (_kernel_dims.height + (_kernel_dims.height - 1) * (_dilation.y() - 1));
        const int            width      = src_x + (_kernel_dims.width + (_kernel_dims.width - 1) * (_dilation.x() - 1));

        for(int y = src_y; y < height; y += _dilation.y())
        {
            for(int x = src_x; x < width; x += _dilation.x(), ++output_ptr)
            {
                if(x < 0 || x >= input_w || y < 0 || y >= input_h)
                {
                    *output_ptr = zero;
                }
                else
                {
                    *output_ptr = *(reinterpret_cast<const T *>(input_ptr + x * input_stride_x + y * input_stride_y));
                }
            }
        }

        if(_has_bias)
        {
            *output_ptr = static_cast<T>(1);
        }
    },
    in, out);
}

NEDepthwiseIm2ColKernel::NEDepthwiseIm2ColKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _kernel_dims(), _conv_info(), _has_bias(), _depth_multiplier(1), _dilation()
{
}

void NEDepthwiseIm2ColKernel::configure(const ITensor *input, ITensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, unsigned int depth_multiplier,
                                        const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), kernel_dims, conv_info, has_bias, depth_multiplier, dilation));

    _input            = input;
    _output           = output;
    _kernel_dims      = kernel_dims;
    _conv_info        = conv_info;
    _has_bias         = has_bias;
    _depth_multiplier = depth_multiplier;
    _dilation         = dilation;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    // Set appropriate function to run
    switch(input->info()->data_type())
    {
        case DataType::QASYMM8:
            _func = &NEDepthwiseIm2ColKernel::run_generic<uint8_t>;
            break;
        case DataType::F16:
            _func = &NEDepthwiseIm2ColKernel::run_generic<half>;
            break;
        case DataType::F32:
            _func = &NEDepthwiseIm2ColKernel::run_generic<float>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }

    // The NEDepthwiseIm2ColKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

Status NEDepthwiseIm2ColKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias, unsigned int depth_multiplier,
                                         const Size2D &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, kernel_dims, conv_info, has_bias, depth_multiplier, dilation));
    return Status{};
}

void NEDepthwiseIm2ColKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if(_func != nullptr)
    {
        (this->*_func)(window);
    }
}
