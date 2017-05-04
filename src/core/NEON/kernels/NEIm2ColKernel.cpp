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
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>

using namespace arm_compute;

void NEIm2ColKernel::run_generic(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const int kernel_depth   = _input->info()->dimension(2);
    const int input_w        = _input->info()->dimension(0);
    const int input_h        = _input->info()->dimension(1);
    const int input_stride_x = _input->info()->strides_in_bytes().x();
    const int input_stride_y = _input->info()->strides_in_bytes().y();
    const int input_stride_z = _input->info()->strides_in_bytes().z();

    int pad_x    = 0;
    int pad_y    = 0;
    int stride_x = 0;
    int stride_y = 0;
    std::tie(pad_x, pad_y)       = _conv_info.pad();
    std::tie(stride_x, stride_y) = _conv_info.stride();

    // Setup input window
    const int start_x = -pad_x;
    const int start_y = -pad_y;

    Window window_in(window);
    // The first three dimensions of the input are increased by the inner loops
    window_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Setup output window
    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, _output->info()->dimension(0), _output->info()->strides_in_bytes().y() / _output->info()->element_size()));
    window_out.set(Window::DimY, Window::Dimension(window.y().start() * _convolved_dims.first, window.y().end() * _convolved_dims.first, _convolved_dims.first));
    window_out.set(Window::DimZ, Window::Dimension(0, 1, 1));

    // Create iterators
    Iterator in(_input, window_in);
    Iterator out(_output, window_out);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int top_left_x = id.x() * stride_x + start_x;
        const int top_left_y = id.y() * stride_y + start_y;

        // Get pointers
        const uint8_t *const input_ptr  = in.ptr();
        auto                 output_ptr = reinterpret_cast<float *>(out.ptr());

        // Linearize volume
        for(int d = 0; d < kernel_depth; ++d)
        {
            for(int y = top_left_y, y_e = top_left_y + static_cast<int>(_kernel_size); y < y_e; ++y)
            {
                for(int x = top_left_x, x_e = top_left_x + static_cast<int>(_kernel_size); x < x_e; ++x, ++output_ptr)
                {
                    if(x < 0 || x >= input_w || y < 0 || y >= input_h)
                    {
                        *output_ptr = 0.f;
                    }
                    else
                    {
                        *output_ptr = *(reinterpret_cast<const float *>(input_ptr + (d * input_stride_z + y * input_stride_y + x * input_stride_x)));
                    }
                }
            }
        }

        // Add bias
        if(_has_bias)
        {
            *output_ptr = 1;
        }
    },
    in, out);
}

void NEIm2ColKernel::run_reduced(const Window &window)
{
    const size_t in_width   = _input->info()->dimension(0);
    const size_t in_height  = _input->info()->dimension(1);
    const size_t out_step_x = in_width * _input->info()->element_size();
    const size_t out_step_y = out_step_x * in_height;
    const size_t out_width  = _output->info()->dimension(0);

    Window in_window(window);
    in_window.set(Window::DimX, Window::Dimension(0, 1, 1));

    Window out_window;
    out_window.use_tensor_dimensions(_output->info());
    out_window.set(Window::DimX, Window::Dimension(out_window.x().start(), out_window.x().end(), in_width));

    Window in_slice  = in_window.first_slice_window_3D();
    Window out_slice = out_window.first_slice_window_1D();

    do
    {
        Iterator in(_input, in_slice);
        Iterator out(_output, out_slice);

        uint8_t *out_ptr = out.ptr();

        execute_window_loop(in_slice, [&](const Coordinates & id)
        {
            memcpy(out_ptr + id.y() * out_step_x + id.z() * out_step_y, in.ptr(), out_step_x);
        },
        in);

        // Add bias
        if(_has_bias)
        {
            *(reinterpret_cast<float *>(out_ptr) + out_width - 1) = 1.0f;
        }
    }
    while(in_window.slide_window_slice_3D(in_slice) && out_window.slide_window_slice_1D(out_slice));
}

NEIm2ColKernel::NEIm2ColKernel()
    : _func(), _input(nullptr), _output(nullptr), _convolved_dims(), _conv_info(), _kernel_size(0), _has_bias(false)
{
}

void NEIm2ColKernel::configure(const ITensor *input, ITensor *output, std::pair<unsigned int, unsigned int> convolved_dims, const PadStrideInfo &conv_info, bool has_bias)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);

    _input          = input;
    _output         = output;
    _convolved_dims = convolved_dims;
    _conv_info      = conv_info;
    _kernel_size    = std::sqrt((output->info()->dimension(0) - (has_bias ? 1 : 0)) / input->info()->dimension(2));
    _has_bias       = has_bias;

    unsigned int pad_x, pad_y, stride_x, stride_y = 0;
    std::tie(pad_x, pad_y)       = conv_info.pad();
    std::tie(stride_x, stride_y) = conv_info.stride();

    bool run_img2col_reduced = (output->info()->dimension(0) == (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))) && (TensorShape::num_max_dimensions >= 4)
                               && (std::equal(input->info()->tensor_shape().cbegin() + 3,
                                              input->info()->tensor_shape().cend(),
                                              output->info()->tensor_shape().cbegin() + 1))
                               && ((stride_x == 1) && (stride_y == 1) && (pad_x == 0) && (pad_y == 0));

    Window window = calculate_max_window(*input->info(), Steps());

    if(run_img2col_reduced)
    {
        _func = &NEIm2ColKernel::run_reduced;
    }
    else
    {
        _func = &NEIm2ColKernel::run_generic;
        window.set(Window::DimX, Window::Dimension(0, _convolved_dims.first, 1));
        window.set(Window::DimY, Window::Dimension(0, _convolved_dims.second, 1));
        window.set(Window::DimZ, Window::Dimension(0, 1, 1));
    }

    // The NEIm2ColKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    IKernel::configure(window);
}

void NEIm2ColKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    (this->*_func)(window);
}
