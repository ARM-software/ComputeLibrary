/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "src/core/NEON/kernels/NEChannelCombineKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEChannelCombineKernel::NEChannelCombineKernel()
    : _func(nullptr), _planes{ { nullptr } }, _output(nullptr), _output_multi(nullptr), _x_subsampling{ { 1, 1, 1 } }, _y_subsampling{ { 1, 1, 1 } }, _num_elems_processed_per_iteration(8),
_is_parallelizable(true)
{
}

void NEChannelCombineKernel::configure(const ITensor *plane0, const ITensor *plane1, const ITensor *plane2, const ITensor *plane3, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(plane0, plane1, plane2, output);
    ARM_COMPUTE_ERROR_ON(plane0 == output);
    ARM_COMPUTE_ERROR_ON(plane1 == output);
    ARM_COMPUTE_ERROR_ON(plane2 == output);

    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane0, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane1, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane2, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::RGB888, Format::RGBA8888, Format::UYVY422, Format::YUYV422);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane0, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane1, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane2, 1, DataType::U8);

    const Format output_format = output->info()->format();

    // Check if horizontal dimension of Y plane is even and validate horizontal sub-sampling dimensions for U and V planes
    if(Format::YUYV422 == output_format || Format::UYVY422 == output_format)
    {
        // Validate Y plane of input and output
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_EVEN(output_format, plane0, output);

        // Validate U and V plane of the input
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(output_format, plane0->info()->tensor_shape(), plane1, plane2);
    }

    _planes[0] = plane0;
    _planes[1] = plane1;
    _planes[2] = plane2;
    _planes[3] = nullptr;

    // Validate the last input tensor only for RGBA format
    if(Format::RGBA8888 == output_format)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(plane3);
        ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane3);

        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane3, Format::U8);
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane3, 1, DataType::U8);

        _planes[3] = plane3;
    }

    _output       = output;
    _output_multi = nullptr;

    // Half the processed elements for U and V channels due to horizontal sub-sampling of 2
    if(Format::YUYV422 == output_format || Format::UYVY422 == output_format)
    {
        _x_subsampling[1] = 2;
        _x_subsampling[2] = 2;
    }

    _num_elems_processed_per_iteration = 8;
    _is_parallelizable                 = true;

    // Select function and number of elements to process given the output format
    switch(output_format)
    {
        case Format::RGB888:
            _func = &NEChannelCombineKernel::combine_3C;
            break;
        case Format::RGBA8888:
            _func = &NEChannelCombineKernel::combine_4C;
            break;
        case Format::UYVY422:
            _num_elems_processed_per_iteration = 16;
            _func                              = &NEChannelCombineKernel::combine_YUV_1p<true>;
            break;
        case Format::YUYV422:
            _num_elems_processed_per_iteration = 16;
            _func                              = &NEChannelCombineKernel::combine_YUV_1p<false>;
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported format.");
            break;
    }

    Window win = calculate_max_window(*plane0->info(), Steps(_num_elems_processed_per_iteration));

    AccessWindowHorizontal output_access(output->info(), 0, _num_elems_processed_per_iteration);
    AccessWindowHorizontal plane0_access(plane0->info(), 0, _num_elems_processed_per_iteration / _x_subsampling[1], 1.f / _x_subsampling[0]);
    AccessWindowHorizontal plane1_access(plane1->info(), 0, _num_elems_processed_per_iteration / _x_subsampling[1], 1.f / _x_subsampling[1]);
    AccessWindowHorizontal plane2_access(plane2->info(), 0, _num_elems_processed_per_iteration / _x_subsampling[1], 1.f / _x_subsampling[2]);
    AccessWindowHorizontal plane3_access(plane3 == nullptr ? nullptr : plane3->info(), 0, _num_elems_processed_per_iteration);

    update_window_and_padding(
        win,
        plane0_access,
        plane1_access,
        plane2_access,
        plane3_access,
        output_access);

    ValidRegion valid_region = intersect_valid_regions(plane0->info()->valid_region(),
                                                       plane1->info()->valid_region(),
                                                       plane2->info()->valid_region());

    if(plane3 != nullptr)
    {
        valid_region = intersect_valid_regions(plane3->info()->valid_region(), valid_region);
    }

    output_access.set_valid_region(win, ValidRegion(valid_region.anchor, output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEChannelCombineKernel::configure(const IImage *plane0, const IImage *plane1, const IImage *plane2, IMultiImage *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(plane0, plane1, plane2, output);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane0);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane1);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(plane2);

    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane0, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane1, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(plane2, Format::U8);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::NV12, Format::NV21, Format::IYUV, Format::YUV444);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane0, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane1, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(plane2, 1, DataType::U8);

    const Format output_format = output->info()->format();

    // Validate shape of Y plane to be even and shape of sub-sampling dimensions for U and V planes
    // Perform validation only for formats which require sub-sampling.
    if(Format::YUV444 != output_format)
    {
        // Validate Y plane of input and output
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_EVEN(output_format, plane0, output->plane(0));

        // Validate U and V plane of the input
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(output_format, plane0->info()->tensor_shape(), plane1, plane2);

        // Validate second plane U (NV12 and NV21 have a UV88 combined plane while IYUV has only the U plane)
        // MultiImage generates the correct tensor shape but also check in case the tensor shape of planes was changed to a wrong size
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(output_format, plane0->info()->tensor_shape(), output->plane(1));

        // Validate the last plane V of format IYUV
        if(Format::IYUV == output_format)
        {
            // Validate Y plane of the output
            ARM_COMPUTE_ERROR_ON_TENSORS_NOT_SUBSAMPLED(output_format, plane0->info()->tensor_shape(), output->plane(2));
        }
    }

    _planes[0]    = plane0;
    _planes[1]    = plane1;
    _planes[2]    = plane2;
    _planes[3]    = nullptr;
    _output       = nullptr;
    _output_multi = output;

    bool         has_two_planes           = false;
    unsigned int num_elems_written_plane1 = 8;

    _num_elems_processed_per_iteration = 8;
    _is_parallelizable                 = true;

    switch(output_format)
    {
        case Format::NV12:
        case Format::NV21:
            _x_subsampling           = { { 1, 2, 2 } };
            _y_subsampling           = { { 1, 2, 2 } };
            _func                    = &NEChannelCombineKernel::combine_YUV_2p;
            has_two_planes           = true;
            num_elems_written_plane1 = 16;
            break;
        case Format::IYUV:
            _is_parallelizable = false;
            _x_subsampling     = { { 1, 2, 2 } };
            _y_subsampling     = { { 1, 2, 2 } };
            _func              = &NEChannelCombineKernel::combine_YUV_3p;
            break;
        case Format::YUV444:
            _is_parallelizable = false;
            _x_subsampling     = { { 1, 1, 1 } };
            _y_subsampling     = { { 1, 1, 1 } };
            _func              = &NEChannelCombineKernel::combine_YUV_3p;
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported format.");
            break;
    }

    const unsigned int y_step = *std::max_element(_y_subsampling.begin(), _y_subsampling.end());

    Window                win = calculate_max_window(*plane0->info(), Steps(_num_elems_processed_per_iteration, y_step));
    AccessWindowRectangle output_plane0_access(output->plane(0)->info(), 0, 0, _num_elems_processed_per_iteration, 1, 1.f, 1.f / _y_subsampling[0]);
    AccessWindowRectangle output_plane1_access(output->plane(1)->info(), 0, 0, num_elems_written_plane1, 1, 1.f / _x_subsampling[1], 1.f / _y_subsampling[1]);
    AccessWindowRectangle output_plane2_access(has_two_planes ? nullptr : output->plane(2)->info(), 0, 0, _num_elems_processed_per_iteration, 1, 1.f / _x_subsampling[2], 1.f / _y_subsampling[2]);

    update_window_and_padding(win,
                              AccessWindowHorizontal(plane0->info(), 0, _num_elems_processed_per_iteration),
                              AccessWindowRectangle(plane1->info(), 0, 0, _num_elems_processed_per_iteration, 1, 1.f / _x_subsampling[1], 1.f / _y_subsampling[1]),
                              AccessWindowRectangle(plane2->info(), 0, 0, _num_elems_processed_per_iteration, 1, 1.f / _x_subsampling[2], 1.f / _y_subsampling[2]),
                              output_plane0_access,
                              output_plane1_access,
                              output_plane2_access);

    ValidRegion plane0_valid_region  = plane0->info()->valid_region();
    ValidRegion output_plane1_region = has_two_planes ? intersect_valid_regions(plane1->info()->valid_region(), plane2->info()->valid_region()) : plane2->info()->valid_region();

    output_plane0_access.set_valid_region(win, ValidRegion(plane0_valid_region.anchor, output->plane(0)->info()->tensor_shape()));
    output_plane1_access.set_valid_region(win, ValidRegion(output_plane1_region.anchor, output->plane(1)->info()->tensor_shape()));
    output_plane2_access.set_valid_region(win, ValidRegion(plane2->info()->valid_region().anchor, output->plane(2)->info()->tensor_shape()));

    INEKernel::configure(win);
}

bool NEChannelCombineKernel::is_parallelisable() const
{
    return _is_parallelizable;
}

void NEChannelCombineKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

void NEChannelCombineKernel::combine_3C(const Window &win)
{
    Iterator p0(_planes[0], win);
    Iterator p1(_planes[1], win);
    Iterator p2(_planes[2], win);
    Iterator out(_output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto p0_ptr  = static_cast<uint8_t *>(p0.ptr());
        const auto p1_ptr  = static_cast<uint8_t *>(p1.ptr());
        const auto p2_ptr  = static_cast<uint8_t *>(p2.ptr());
        const auto out_ptr = static_cast<uint8_t *>(out.ptr());

        const uint8x8x3_t pixels =
        {
            {
                vld1_u8(p0_ptr),
                vld1_u8(p1_ptr),
                vld1_u8(p2_ptr)
            }
        };

        vst3_u8(out_ptr, pixels);
    },
    p0, p1, p2, out);
}

void NEChannelCombineKernel::combine_4C(const Window &win)
{
    Iterator p0(_planes[0], win);
    Iterator p1(_planes[1], win);
    Iterator p2(_planes[2], win);
    Iterator p3(_planes[3], win);
    Iterator out(_output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto p0_ptr  = static_cast<uint8_t *>(p0.ptr());
        const auto p1_ptr  = static_cast<uint8_t *>(p1.ptr());
        const auto p2_ptr  = static_cast<uint8_t *>(p2.ptr());
        const auto p3_ptr  = static_cast<uint8_t *>(p3.ptr());
        const auto out_ptr = static_cast<uint8_t *>(out.ptr());

        const uint8x8x4_t pixels =
        {
            {
                vld1_u8(p0_ptr),
                vld1_u8(p1_ptr),
                vld1_u8(p2_ptr),
                vld1_u8(p3_ptr)
            }
        };

        vst4_u8(out_ptr, pixels);
    },
    p0, p1, p2, p3, out);
}

template <bool is_uyvy>
void NEChannelCombineKernel::combine_YUV_1p(const Window &win)
{
    // Create sub-sampled uv window and init uv planes
    Window win_uv(win);
    win_uv.set_dimension_step(Window::DimX, win.x().step() / _x_subsampling[1]);
    win_uv.validate();

    Iterator p0(_planes[0], win);
    Iterator p1(_planes[1], win_uv);
    Iterator p2(_planes[2], win_uv);
    Iterator out(_output, win);

    constexpr auto shift = is_uyvy ? 1 : 0;

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto p0_ptr  = static_cast<uint8_t *>(p0.ptr());
        const auto p1_ptr  = static_cast<uint8_t *>(p1.ptr());
        const auto p2_ptr  = static_cast<uint8_t *>(p2.ptr());
        const auto out_ptr = static_cast<uint8_t *>(out.ptr());

        const uint8x8x2_t pixels_y = vld2_u8(p0_ptr);
        const uint8x8x2_t pixels_uv =
        {
            {
                vld1_u8(p1_ptr),
                vld1_u8(p2_ptr)
            }
        };

        uint8x8x4_t pixels{ {} };
        pixels.val[0 + shift] = pixels_y.val[0];
        pixels.val[1 - shift] = pixels_uv.val[0];
        pixels.val[2 + shift] = pixels_y.val[1];
        pixels.val[3 - shift] = pixels_uv.val[1];

        vst4_u8(out_ptr, pixels);
    },
    p0, p1, p2, out);
}

void NEChannelCombineKernel::combine_YUV_2p(const Window &win)
{
    ARM_COMPUTE_ERROR_ON(win.x().start() % _x_subsampling[1]);
    ARM_COMPUTE_ERROR_ON(win.y().start() % _y_subsampling[1]);

    // Copy first plane
    copy_plane(win, 0);

    // Update UV window
    Window uv_win(win);
    uv_win.set(Window::DimX, Window::Dimension(uv_win.x().start() / _x_subsampling[1], uv_win.x().end() / _x_subsampling[1], uv_win.x().step() / _x_subsampling[1]));
    uv_win.set(Window::DimY, Window::Dimension(uv_win.y().start() / _y_subsampling[1], uv_win.y().end() / _y_subsampling[1], 1));
    uv_win.validate();

    // Update output win
    Window out_win(win);
    out_win.set(Window::DimX, Window::Dimension(out_win.x().start(), out_win.x().end(), out_win.x().step() / _x_subsampling[1]));
    out_win.set(Window::DimY, Window::Dimension(out_win.y().start() / _y_subsampling[1], out_win.y().end() / _y_subsampling[1], 1));
    out_win.validate();

    // Construct second plane
    const int shift = (Format::NV12 == _output_multi->info()->format()) ? 0 : 1;
    Iterator  p1(_planes[1 + shift], uv_win);
    Iterator  p2(_planes[2 - shift], uv_win);
    Iterator  out(_output_multi->plane(1), out_win);

    // Increase step size after iterator is created to calculate stride correctly for multi channel format
    out_win.set_dimension_step(Window::DimX, out_win.x().step() * _x_subsampling[1]);

    execute_window_loop(out_win, [&](const Coordinates &)
    {
        const uint8x8x2_t pixels =
        {
            {
                vld1_u8(p1.ptr()),
                vld1_u8(p2.ptr())
            }
        };

        vst2_u8(out.ptr(), pixels);
    },
    p1, p2, out);
}

void NEChannelCombineKernel::combine_YUV_3p(const Window &win)
{
    copy_plane(win, 0);
    copy_plane(win, 1);
    copy_plane(win, 2);
}

void NEChannelCombineKernel::copy_plane(const Window &win, uint32_t plane_id)
{
    ARM_COMPUTE_ERROR_ON(win.x().start() % _x_subsampling[plane_id]);
    ARM_COMPUTE_ERROR_ON(win.y().start() % _y_subsampling[plane_id]);

    // Update window
    Window tmp_win(win);
    tmp_win.set(Window::DimX, Window::Dimension(tmp_win.x().start() / _x_subsampling[plane_id], tmp_win.x().end() / _x_subsampling[plane_id], tmp_win.x().step() / _x_subsampling[plane_id]));
    tmp_win.set(Window::DimY, Window::Dimension(tmp_win.y().start() / _y_subsampling[plane_id], tmp_win.y().end() / _y_subsampling[plane_id], 1));

    Iterator in(_planes[plane_id], tmp_win);
    Iterator out(_output_multi->plane(plane_id), tmp_win);

    execute_window_loop(tmp_win, [&](const Coordinates &)
    {
        const uint8x8_t pixels = vld1_u8(in.ptr());

        vst1_u8(out.ptr(), pixels);
    },
    in, out);
}
