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
#include "arm_compute/core/NEON/kernels/NEDepthwiseConvolution3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/convolution/NEDirectConvolutionDetail.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/AccessWindowTranspose.h"
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
using namespace arm_compute::detail;

NEDepthwiseConvolution3x3Kernel::NEDepthwiseConvolution3x3Kernel()
    : _border_size(0), _input(), _output(), _weights(), _conv_info()
{
}

BorderSize NEDepthwiseConvolution3x3Kernel::border_size() const
{
    return _border_size;
}

void NEDepthwiseConvolution3x3Kernel::configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, weights);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != 3 || weights->info()->dimension(1) != 3);

    std::pair<unsigned int, unsigned int> expected_output = scaled_dimensions(input->info()->tensor_shape().x(), input->info()->tensor_shape().y(),
                                                                              weights->info()->tensor_shape().x(), weights->info()->tensor_shape().y(),
                                                                              conv_info);

    ARM_COMPUTE_UNUSED(expected_output);
    ARM_COMPUTE_ERROR_ON(expected_output.first != output->info()->tensor_shape().x());
    ARM_COMPUTE_ERROR_ON(expected_output.second != output->info()->tensor_shape().y());

    _input                           = input;
    _output                          = output;
    _weights                         = weights;
    _conv_info                       = conv_info;
    const unsigned int conv_stride_x = conv_info.stride().first;
    const unsigned int conv_pad_x    = conv_info.pad().first;
    const unsigned int conv_pad_y    = conv_info.pad().second;

    ARM_COMPUTE_ERROR_ON(conv_stride_x < 1 || conv_stride_x > 3);

    const unsigned int num_elems_written_per_iteration = 16 >> conv_stride_x;
    _border_size                                       = BorderSize(conv_pad_y, conv_pad_x);

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_written_per_iteration));

    AccessWindowStatic     input_access(input->info(), -conv_pad_x, -conv_pad_y, input->info()->dimension(0) + _border_size.right, input->info()->dimension(1) + _border_size.bottom);
    AccessWindowStatic     weights_access(weights->info(), 0, 0, weights->info()->dimension(0), weights->info()->dimension(1));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, input_access, weights_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

template <unsigned int stridex>
class convolver_3x3
{
public:
    static void convolve(const Window &window, unsigned int num_elems_written_per_iteration,
                         const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info)
    {
        const int          input_stride_x  = input->info()->strides_in_bytes().x();
        const int          input_stride_y  = input->info()->strides_in_bytes().y();
        const int          output_stride_y = output->info()->strides_in_bytes().y();
        const int          kernel_stride_y = weights->info()->strides_in_bytes().y();
        const int          kernel_stride_z = weights->info()->strides_in_bytes().z();
        const int          output_w        = output->info()->dimension(0);
        const int          output_h        = output->info()->dimension(1);
        const int          delta_input     = get_input_num_elems_processed<stridex>(num_elems_written_per_iteration);
        const unsigned int conv_stride_y   = std::get<1>(conv_info.stride());
        const unsigned int conv_pad_x      = std::get<0>(conv_info.pad());
        const unsigned int conv_pad_y      = std::get<1>(conv_info.pad());

        // setup output window for the iterator
        Window window_out = window;
        window_out.set(Window::DimX, Window::Dimension(0, output->info()->dimension(Window::DimX), output->info()->dimension(Window::DimX)));
        window_out.set(Window::DimY, Window::Dimension(0, output->info()->dimension(Window::DimY), output->info()->dimension(Window::DimY)));

        // setup input window for the iterator
        Window window_in = window;
        // we just want execute_window_loop to iterate over the dimensions > 2, so we set the first 2 dimensions to 0
        window_in.set(Window::DimX, Window::Dimension(0, 0, 0));
        window_in.set(Window::DimY, Window::Dimension(0, 0, 0));

        Window window_k = calculate_max_window(*weights->info(), Steps(1u));

        Iterator in(input, window_in);
        Iterator out(output, window_out);
        Iterator w(weights, window_k);

        const uint8_t *weights_ptr = w.ptr();

        execute_window_loop(window_out, [&](const Coordinates & id)
        {
            const uint8_t *input_ptr = in.ptr() - conv_pad_x * input_stride_x - conv_pad_y * input_stride_y;
            int            ih        = 0;
            int            oh        = 0;

            const uint8_t *ptr_weights_base = weights_ptr + id.z() * kernel_stride_z;
            const auto     ptr_weights_r0   = reinterpret_cast<const float *>(ptr_weights_base);
            const auto     ptr_weights_r1   = reinterpret_cast<const float *>(ptr_weights_base + kernel_stride_y);
            const auto     ptr_weights_r2   = reinterpret_cast<const float *>(ptr_weights_base + kernel_stride_y * 2);
            const auto     vw_r0            = load_matrix_row(ptr_weights_r0);
            const auto     vw_r1            = load_matrix_row(ptr_weights_r1);
            const auto     vw_r2            = load_matrix_row(ptr_weights_r2);

            for(ih = 0, oh = 0; oh < output_h; ++oh, ih += conv_stride_y)
            {
                auto in_top = reinterpret_cast<const float *>(input_ptr + (ih + 0) * input_stride_y);
                auto in_mid = reinterpret_cast<const float *>(input_ptr + (ih + 1) * input_stride_y);
                auto in_low = reinterpret_cast<const float *>(input_ptr + (ih + 2) * input_stride_y);
                auto p_out  = reinterpret_cast<float *>(out.ptr() + oh * output_stride_y);

                for(int ow = 0; ow < output_w; ow += num_elems_written_per_iteration,
                    in_top += delta_input, in_mid += delta_input, in_low += delta_input, p_out += num_elems_written_per_iteration)
                {
                    auto vres = convolve_3x3<stridex>(in_top, in_mid, in_low, vw_r0, vw_r1, vw_r2, 0);
                    store_results<stridex>(p_out, vres);
                }
            }
        },
        in, out);
    }
};

void NEDepthwiseConvolution3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_UNUSED(info);

    const unsigned int conv_stride_x                   = _conv_info.stride().first;
    const unsigned int num_elems_written_per_iteration = 16 >> conv_stride_x;

    switch(conv_stride_x)
    {
        case 1:
            convolver_3x3<1>::convolve(window, num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            break;
        case 2:
            convolver_3x3<2>::convolve(window, num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            break;
        case 3:
            convolver_3x3<3>::convolve(window, num_elems_written_per_iteration, _input, _weights, _output, _conv_info);
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }
}
