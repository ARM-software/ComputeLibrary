/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEWarpKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cstddef>

using namespace arm_compute;

namespace
{
inline uint8_t nearest_interpolation(const uint8_t *in_ptr, int x, int y, size_t stride)
{
    return in_ptr[x + y * stride];
}
} // namespace

INEWarpKernel::INEWarpKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _constant_border_value(0), _matrix(nullptr)
{
}

BorderSize INEWarpKernel::border_size() const
{
    return BorderSize(1);
}

void INEWarpKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

void INEWarpKernel::configure(const ITensor *input, ITensor *output, const float *matrix, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == matrix);

    _matrix                = matrix;
    _constant_border_value = constant_border_value;

    switch(border_mode)
    {
        case BorderMode::UNDEFINED:
            _func = &INEWarpKernel::warp_undefined;
            break;
        case BorderMode::CONSTANT:
            _func = &INEWarpKernel::warp_constant;
            break;
        case BorderMode::REPLICATE:
            _func = &INEWarpKernel::warp_replicate;
            break;
        default:
            ARM_COMPUTE_ERROR("Border mode not supported");
            break;
    }

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(1U));

    const ValidRegion &input_valid_region = input->info()->valid_region();

    // Reads can occur within the valid region of the input
    AccessWindowStatic input_access(input->info(),
                                    input_valid_region.anchor[0] - border_size().left, input_valid_region.anchor[1] - border_size().top,
                                    input_valid_region.anchor[0] + input_valid_region.shape[0] + border_size().right,
                                    input_valid_region.anchor[1] + input_valid_region.shape[1] + border_size().bottom);
    AccessWindowHorizontal output_access(output->info(), 0, 1);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

template <InterpolationPolicy interpolation>
void NEWarpAffineKernel<interpolation>::warp_undefined(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    const int    min_x  = _input->info()->valid_region().anchor[0];
    const int    max_x  = min_x + _input->info()->valid_region().shape[0];
    const int    min_y  = _input->info()->valid_region().anchor[1];
    const int    max_y  = min_y + _input->info()->valid_region().shape[1];
    const size_t stride = _input->info()->strides_in_bytes()[1];

    // x0 = M01 * x + M01 * y + M02
    // y0 = M11 * x + M11 * y + M12
    const float M00 = _matrix[0];
    const float M10 = _matrix[1];
    const float M01 = _matrix[0 + 1 * 2];
    const float M11 = _matrix[1 + 1 * 2];
    const float M02 = _matrix[0 + 2 * 2];
    const float M12 = _matrix[1 + 2 * 2];

    // "M00 * x" and "M10 * x", when x = window.x.start
    const float start_x0 = M00 * window.x().start();
    const float start_y0 = M10 * window.x().start();

    // Current row
    int y_cur  = window.y().start();
    int z_cur  = window.z().start();
    int d3_cur = window[3].start();
    int d4_cur = window[4].start();
    int d5_cur = window[5].start();

    // const_x0 and const_y0 are the constant parts of x0 and y0 during the row processing
    float const_x0 = M01 * y_cur + M02;
    float const_y0 = M11 * y_cur + M12;

    // Affine warp coordinates
    float x0 = start_x0 + const_x0;
    float y0 = start_y0 + const_y0;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Check if we are processing a new row. If so, update the current processed row (y_cur), x0, y0 and z0
        if((y_cur != id.y()) || (z_cur != id.z()) || (d3_cur != id[3]) || (d4_cur != id[4]) || (d5_cur != id[5]))
        {
            y_cur  = id.y();
            z_cur  = id.z();
            d3_cur = id[3];
            d4_cur = id[4];
            d5_cur = id[5];

            const_x0 = M01 * y_cur + M02;
            const_y0 = M11 * y_cur + M12;

            x0 = start_x0 + const_x0;
            y0 = start_y0 + const_y0;
        }

        // Only write to output if x0 and y0 are within the valid region.
        // Otherwise the read value would be undefined.
        if((min_y <= y0) && (y0 < max_y) && (min_x <= x0) && (x0 < max_x))
        {
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = nearest_interpolation(in.ptr(), x0, y0, stride);
                    break;
                case InterpolationPolicy::BILINEAR:
                    *out.ptr() = pixel_bilinear_c1(in.ptr(), stride, x0, y0);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }

        x0 += M00;
        y0 += M10;
    },
    in, out);
}

template <InterpolationPolicy interpolation>
void NEWarpAffineKernel<interpolation>::warp_constant(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    const int    min_x  = _input->info()->valid_region().anchor[0];
    const int    max_x  = min_x + _input->info()->valid_region().shape[0];
    const int    min_y  = _input->info()->valid_region().anchor[1];
    const int    max_y  = min_y + _input->info()->valid_region().shape[1];
    const size_t stride = _input->info()->strides_in_bytes()[1];

    // x0 = M01 * x + M01 * y + M02
    // y0 = M11 * x + M11 * y + M12
    const float M00 = _matrix[0];
    const float M10 = _matrix[1];
    const float M01 = _matrix[0 + 1 * 2];
    const float M11 = _matrix[1 + 1 * 2];
    const float M02 = _matrix[0 + 2 * 2];
    const float M12 = _matrix[1 + 2 * 2];

    // "M00 * x" and "M10 * x", when x = window.x.start
    const float start_x0 = M00 * window.x().start();
    const float start_y0 = M10 * window.x().start();

    // Current row
    int y_cur  = window.y().start();
    int z_cur  = window.z().start();
    int d3_cur = window[3].start();
    int d4_cur = window[4].start();
    int d5_cur = window[5].start();

    // const_x0 and const_y0 are the constant parts of x0 and y0 during the row processing
    float const_x0 = M01 * y_cur + M02;
    float const_y0 = M11 * y_cur + M12;

    // Affine warp coordinates
    float x0 = start_x0 + const_x0;
    float y0 = start_y0 + const_y0;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Check if we are processing a new row. If so, update the current processed row (y_cur), x0, y0 and z0
        if((y_cur != id.y()) || (z_cur != id.z()) || (d3_cur != id[3]) || (d4_cur != id[4]) || (d5_cur != id[5]))
        {
            y_cur  = id.y();
            z_cur  = id.z();
            d3_cur = id[3];
            d4_cur = id[4];
            d5_cur = id[5];

            const_x0 = M01 * y_cur + M02;
            const_y0 = M11 * y_cur + M12;

            x0 = start_x0 + const_x0;
            y0 = start_y0 + const_y0;
        }

        // Only use input values if x0 and y0 are within the valid region.
        // Otherwise write the constant border value.
        if((min_y <= y0) && (y0 < max_y) && (min_x <= x0) && (x0 < max_x))
        {
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = nearest_interpolation(in.ptr(), x0, y0, stride);
                    break;
                case InterpolationPolicy::BILINEAR:
                    *out.ptr() = pixel_bilinear_c1(in.ptr(), stride, x0, y0);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }
        else
        {
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = _constant_border_value;
                    break;
                case InterpolationPolicy::BILINEAR:
                {
                    const auto xi   = utility::clamp<int>(std::floor(x0), min_x - 1, max_x);
                    const auto yi   = utility::clamp<int>(std::floor(y0), min_y - 1, max_y);
                    const auto xi_1 = utility::clamp<int>(std::floor(x0 + 1), min_x - 1, max_x);
                    const auto yi_1 = utility::clamp<int>(std::floor(y0 + 1), min_y - 1, max_y);

                    const float dx  = x0 - std::floor(x0);
                    const float dy  = y0 - std::floor(y0);
                    const float dx1 = 1.0f - dx;
                    const float dy1 = 1.0f - dy;

                    const float a00 = *(in.ptr() + xi + yi * stride);
                    const float a01 = *(in.ptr() + xi_1 + yi * stride);
                    const float a10 = *(in.ptr() + xi + yi_1 * stride);
                    const float a11 = *(in.ptr() + xi_1 + yi_1 * stride);

                    *out.ptr() = a00 * (dx1 * dy1) + a01 * (dx * dy1) + a10 * (dx1 * dy) + a11 * (dx * dy);
                }
                break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }

        x0 += M00;
        y0 += M10;
    },
    in, out);
}

template <InterpolationPolicy interpolation>
void NEWarpAffineKernel<interpolation>::warp_replicate(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    const int    min_x  = _input->info()->valid_region().anchor[0];
    const int    max_x  = min_x + _input->info()->valid_region().shape[0];
    const int    min_y  = _input->info()->valid_region().anchor[1];
    const int    max_y  = min_y + _input->info()->valid_region().shape[1];
    const size_t stride = _input->info()->strides_in_bytes()[1];

    // Current row
    int y_cur  = window.y().start();
    int z_cur  = window.z().start();
    int d3_cur = window[3].start();
    int d4_cur = window[4].start();
    int d5_cur = window[5].start();

    const float M00 = _matrix[0];
    const float M10 = _matrix[1];
    const float M01 = _matrix[0 + 1 * 2];
    const float M11 = _matrix[1 + 1 * 2];
    const float M02 = _matrix[0 + 2 * 2];
    const float M12 = _matrix[1 + 2 * 2];

    // "M00 * x" and "M10 * x", when x = window.x.start
    const float start_x0 = M00 * window.x().start();
    const float start_y0 = M10 * window.x().start();

    // const_x0 and const_y0 are the constant parts of x0 and y0 during the row processing
    float const_x0 = M01 * y_cur + M02;
    float const_y0 = M11 * y_cur + M12;

    float x0 = start_x0 + const_x0;
    float y0 = start_y0 + const_y0;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Check if we are processing a new row. If so, update the current processed row (y_cur), x0, y0 and z0
        if((y_cur != id.y()) || (z_cur != id.z()) || (d3_cur != id[3]) || (d4_cur != id[4]) || (d5_cur != id[5]))
        {
            y_cur  = id.y();
            z_cur  = id.z();
            d3_cur = id[3];
            d4_cur = id[4];
            d5_cur = id[5];

            const_x0 = M01 * y_cur + M02;
            const_y0 = M11 * y_cur + M12;

            x0 = start_x0 + const_x0;
            y0 = start_y0 + const_y0;
        }

        // Only load from (x0, y0) if the point is within the valid region.
        // Otherwise load from the edge of the valid region.
        if((min_y <= y0) && (y0 < max_y) && (min_x <= x0) && (x0 < max_x))
        {
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = nearest_interpolation(in.ptr(), x0, y0, stride);
                    break;
                case InterpolationPolicy::BILINEAR:
                    *out.ptr() = pixel_bilinear_c1(in.ptr(), stride, x0, y0);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }
        else
        {
            // Clamp coordinates
            const auto xi = utility::clamp<int>(std::floor(x0), min_x, max_x - 1);
            const auto yi = utility::clamp<int>(std::floor(y0), min_y, max_y - 1);
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = *(in.ptr() + xi + yi * stride);
                    break;
                case InterpolationPolicy::BILINEAR:
                {
                    const auto xi_1 = utility::clamp<int>(std::floor(x0 + 1), min_x, max_x - 1);
                    const auto yi_1 = utility::clamp<int>(std::floor(y0 + 1), min_y, max_y - 1);

                    const float dx  = x0 - std::floor(x0);
                    const float dy  = y0 - std::floor(y0);
                    const float dx1 = 1.0f - dx;
                    const float dy1 = 1.0f - dy;

                    const float a00 = *(in.ptr() + xi + yi * stride);
                    const float a01 = *(in.ptr() + xi_1 + yi * stride);
                    const float a10 = *(in.ptr() + xi + yi_1 * stride);
                    const float a11 = *(in.ptr() + xi_1 + yi_1 * stride);

                    *out.ptr() = a00 * (dx1 * dy1) + a01 * (dx * dy1) + a10 * (dx1 * dy) + a11 * (dx * dy);
                }
                break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }

        x0 += M00;
        y0 += M10;
    },
    in, out);
}

template <InterpolationPolicy interpolation>
void NEWarpPerspectiveKernel<interpolation>::warp_undefined(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    const int    min_x  = _input->info()->valid_region().anchor[0];
    const int    max_x  = min_x + _input->info()->valid_region().shape[0];
    const int    min_y  = _input->info()->valid_region().anchor[1];
    const int    max_y  = min_y + _input->info()->valid_region().shape[1];
    const size_t stride = _input->info()->strides_in_bytes()[1];

    // x0 = M00 * x + M01 * y + M02
    // y0 = M10 * x + M11 * y + M12
    // z0 = M20 * x + M21 * y + M22
    // xn = x0 / z0
    // yn = y0 / z0
    const float M00 = _matrix[0];
    const float M10 = _matrix[1];
    const float M20 = _matrix[2];
    const float M01 = _matrix[0 + 1 * 3];
    const float M11 = _matrix[1 + 1 * 3];
    const float M21 = _matrix[2 + 1 * 3];
    const float M02 = _matrix[0 + 2 * 3];
    const float M12 = _matrix[1 + 2 * 3];
    const float M22 = _matrix[2 + 2 * 3];

    // "M00 * x", "M10 * x" and "M20 * x", when x = window.x.start
    const float start_x0 = M00 * window.x().start();
    const float start_y0 = M10 * window.x().start();
    const float start_z0 = M20 * window.x().start();

    // Current row
    int y_cur  = window.y().start();
    int z_cur  = window.z().start();
    int d3_cur = window[3].start();
    int d4_cur = window[4].start();
    int d5_cur = window[5].start();

    // const_x0, const_y0 and const_z0 are the constant parts of x0, y0 and z0 during the row processing
    float const_x0 = M01 * y_cur + M02;
    float const_y0 = M11 * y_cur + M12;
    float const_z0 = M21 * y_cur + M22;

    // Perspective warp coordinates
    float x0 = start_x0 + const_x0;
    float y0 = start_y0 + const_y0;
    float z0 = start_z0 + const_z0;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Check if we are processing a new row. If so, update the current processed row (y_cur), x0, y0 and z0
        if((y_cur != id.y()) || (z_cur != id.z()) || (d3_cur != id[3]) || (d4_cur != id[4]) || (d5_cur != id[5]))
        {
            y_cur  = id.y();
            z_cur  = id.z();
            d3_cur = id[3];
            d4_cur = id[4];
            d5_cur = id[5];

            const_x0 = M01 * y_cur + M02;
            const_y0 = M11 * y_cur + M12;
            const_z0 = M21 * y_cur + M22;

            x0 = start_x0 + const_x0;
            y0 = start_y0 + const_y0;
            z0 = start_z0 + const_z0;
        }

        const float xn = x0 / z0;
        const float yn = y0 / z0;

        // Only write to output if xn and yn are within the valid region.
        // Otherwise the read value would be undefined.
        if((min_y <= yn) && (yn < max_y) && (min_x <= xn) && (xn < max_x))
        {
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = nearest_interpolation(in.ptr(), xn, yn, stride);
                    break;
                case InterpolationPolicy::BILINEAR:
                    *out.ptr() = pixel_bilinear_c1(in.ptr(), stride, xn, yn);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }

        x0 += M00;
        y0 += M10;
        z0 += M20;
    },
    in, out);
}

template <InterpolationPolicy interpolation>
void NEWarpPerspectiveKernel<interpolation>::warp_constant(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    const int    min_x  = _input->info()->valid_region().anchor[0];
    const int    max_x  = min_x + _input->info()->valid_region().shape[0];
    const int    min_y  = _input->info()->valid_region().anchor[1];
    const int    max_y  = min_y + _input->info()->valid_region().shape[1];
    const size_t stride = _input->info()->strides_in_bytes()[1];

    // x0 = M00 * x + M01 * y + M02
    // y0 = M10 * x + M11 * y + M12
    // z0 = M20 * x + M21 * y + M22
    // xn = x0 / z0
    // yn = y0 / z0
    const float M00 = _matrix[0];
    const float M10 = _matrix[1];
    const float M20 = _matrix[2];
    const float M01 = _matrix[0 + 1 * 3];
    const float M11 = _matrix[1 + 1 * 3];
    const float M21 = _matrix[2 + 1 * 3];
    const float M02 = _matrix[0 + 2 * 3];
    const float M12 = _matrix[1 + 2 * 3];
    const float M22 = _matrix[2 + 2 * 3];

    // "M00 * x", "M10 * x" and "M20 * x", when x = window.x.start
    const float start_x0 = M00 * window.x().start();
    const float start_y0 = M10 * window.x().start();
    const float start_z0 = M20 * window.x().start();

    // Current row
    int y_cur  = window.y().start();
    int z_cur  = window.z().start();
    int d3_cur = window[3].start();
    int d4_cur = window[4].start();
    int d5_cur = window[5].start();

    // const_x0, const_y0 and const_z0 are the constant parts of x0, y0 and z0 during the row processing
    float const_x0 = M01 * y_cur + M02;
    float const_y0 = M11 * y_cur + M12;
    float const_z0 = M21 * y_cur + M22;

    // Perspective warp coordinates
    float x0 = start_x0 + const_x0;
    float y0 = start_y0 + const_y0;
    float z0 = start_z0 + const_z0;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Check if we are processing a new row. If so, update the current processed row (y_cur), x0, y0 and z0
        if((y_cur != id.y()) || (z_cur != id.z()) || (d3_cur != id[3]) || (d4_cur != id[4]) || (d5_cur != id[5]))
        {
            y_cur  = id.y();
            z_cur  = id.z();
            d3_cur = id[3];
            d4_cur = id[4];
            d5_cur = id[5];

            const_x0 = M01 * y_cur + M02;
            const_y0 = M11 * y_cur + M12;
            const_z0 = M21 * y_cur + M22;

            x0 = start_x0 + const_x0;
            y0 = start_y0 + const_y0;
            z0 = start_z0 + const_z0;
        }

        const float xn = x0 / z0;
        const float yn = y0 / z0;

        // Only use input values if xn and yn are within the valid region.
        if((min_y <= yn) && (yn < max_y) && (min_x <= xn) && (xn < max_x))
        {
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = nearest_interpolation(in.ptr(), xn, yn, stride);
                    break;
                case InterpolationPolicy::BILINEAR:
                    *out.ptr() = pixel_bilinear_c1(in.ptr(), stride, xn, yn);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }
        else
        {
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = _constant_border_value;
                    break;
                case InterpolationPolicy::BILINEAR:
                {
                    const auto xi   = utility::clamp<int>(std::floor(xn), min_x - 1, max_x);
                    const auto yi   = utility::clamp<int>(std::floor(yn), min_y - 1, max_y);
                    const auto xi_1 = utility::clamp<int>(std::floor(xn + 1), min_x - 1, max_x);
                    const auto yi_1 = utility::clamp<int>(std::floor(yn + 1), min_y - 1, max_y);

                    const float dx  = xn - std::floor(xn);
                    const float dy  = yn - std::floor(yn);
                    const float dx1 = 1.0f - dx;
                    const float dy1 = 1.0f - dy;

                    const float a00 = *(in.ptr() + xi + yi * stride);
                    const float a01 = *(in.ptr() + xi_1 + yi * stride);
                    const float a10 = *(in.ptr() + xi + yi_1 * stride);
                    const float a11 = *(in.ptr() + xi_1 + yi_1 * stride);

                    *out.ptr() = a00 * (dx1 * dy1) + a01 * (dx * dy1) + a10 * (dx1 * dy) + a11 * (dx * dy);
                }
                break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }

        x0 += M00;
        y0 += M10;
        z0 += M20;
    },
    in, out);
}

template <InterpolationPolicy interpolation>
void NEWarpPerspectiveKernel<interpolation>::warp_replicate(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    const int    min_x  = _input->info()->valid_region().anchor[0];
    const int    max_x  = min_x + _input->info()->valid_region().shape[0];
    const int    min_y  = _input->info()->valid_region().anchor[1];
    const int    max_y  = min_y + _input->info()->valid_region().shape[1];
    const size_t stride = _input->info()->strides_in_bytes()[1];

    // Current row
    int y_cur  = window.y().start();
    int z_cur  = window.z().start();
    int d3_cur = window[3].start();
    int d4_cur = window[4].start();
    int d5_cur = window[5].start();

    // x0 = M00 * x + M01 * y + M02
    // y0 = M10 * x + M11 * y + M12
    // z0 = M20 * x + M21 * y + M22
    // xn = x0 / z0
    // yn = y0 / z0
    const float M00 = _matrix[0];
    const float M10 = _matrix[1];
    const float M20 = _matrix[2];
    const float M01 = _matrix[0 + 1 * 3];
    const float M11 = _matrix[1 + 1 * 3];
    const float M21 = _matrix[2 + 1 * 3];
    const float M02 = _matrix[0 + 2 * 3];
    const float M12 = _matrix[1 + 2 * 3];
    const float M22 = _matrix[2 + 2 * 3];

    // "M00 * x", "M10 * x" and "M20 * x", when x = window.x.start
    const float start_x0 = M00 * window.x().start();
    const float start_y0 = M10 * window.x().start();
    const float start_z0 = M20 * window.x().start();

    // const_x0, const_y0 and const_z0 are the constant parts of x0, y0 and z0 during the row processing
    float const_x0 = M01 * y_cur + M02;
    float const_y0 = M11 * y_cur + M12;
    float const_z0 = M21 * y_cur + M22;

    // Perspective warp coordinates
    float x0 = start_x0 + const_x0;
    float y0 = start_y0 + const_y0;
    float z0 = start_z0 + const_z0;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Check if we are processing a new row. If so, update the current processed row (y_cur), x0, y0 and z0
        if((y_cur != id.y()) || (z_cur != id.z()) || (d3_cur != id[3]) || (d4_cur != id[4]) || (d5_cur != id[5]))
        {
            y_cur  = id.y();
            z_cur  = id.z();
            d3_cur = id[3];
            d4_cur = id[4];
            d5_cur = id[5];

            const_x0 = M01 * y_cur + M02;
            const_y0 = M11 * y_cur + M12;
            const_z0 = M21 * y_cur + M22;

            x0 = start_x0 + const_x0;
            y0 = start_y0 + const_y0;
            z0 = start_z0 + const_z0;
        }

        const float xn = x0 / z0;
        const float yn = y0 / z0;

        // Only load from (x0, y0) if the point is within the valid region.
        if((min_y <= yn) && (yn < max_y) && (min_x <= xn) && (xn < max_x))
        {
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = nearest_interpolation(in.ptr(), xn, yn, stride);
                    break;
                case InterpolationPolicy::BILINEAR:
                    *out.ptr() = pixel_bilinear_c1(in.ptr(), stride, xn, yn);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }
        else
        {
            // Clamp coordinates
            const auto xi = utility::clamp<int>(std::floor(xn), min_x, max_x - 1);
            const auto yi = utility::clamp<int>(std::floor(yn), min_y, max_y - 1);
            switch(interpolation)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    *out.ptr() = *(in.ptr() + xi + yi * stride);
                    break;
                case InterpolationPolicy::BILINEAR:
                {
                    const auto xi_1 = utility::clamp<int>(std::floor(xn + 1), min_x, max_x - 1);
                    const auto yi_1 = utility::clamp<int>(std::floor(yn + 1), min_y, max_y - 1);

                    const float dx  = xn - std::floor(xn);
                    const float dy  = yn - std::floor(yn);
                    const float dx1 = 1.0f - dx;
                    const float dy1 = 1.0f - dy;

                    const float a00 = *(in.ptr() + xi + yi * stride);
                    const float a01 = *(in.ptr() + xi_1 + yi * stride);
                    const float a10 = *(in.ptr() + xi + yi_1 * stride);
                    const float a11 = *(in.ptr() + xi_1 + yi_1 * stride);

                    *out.ptr() = a00 * (dx1 * dy1) + a01 * (dx * dy1) + a10 * (dx1 * dy) + a11 * (dx * dy);
                }
                break;
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
            }
        }

        x0 += M00;
        y0 += M10;
        z0 += M20;
    },
    in, out);
}

template class arm_compute::NEWarpAffineKernel<InterpolationPolicy::NEAREST_NEIGHBOR>;
template class arm_compute::NEWarpAffineKernel<InterpolationPolicy::BILINEAR>;
template class arm_compute::NEWarpPerspectiveKernel<InterpolationPolicy::NEAREST_NEIGHBOR>;
template class arm_compute::NEWarpPerspectiveKernel<InterpolationPolicy::BILINEAR>;
