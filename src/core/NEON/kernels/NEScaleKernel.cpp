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
#include "arm_compute/core/NEON/kernels/NEScaleKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/Rounding.h"
#include "arm_compute/core/utils/misc/Utility.h"

#include "src/core/utils/ScaleUtils.h"

#include <arm_neon.h>
#include <map>

namespace arm_compute
{
namespace
{
inline float compute_bilinear(float a00, float a01, float a10, float a11, float dx_val, float dy_val)
{
    const float dx1_val = 1.0f - dx_val;
    const float dy1_val = 1.0f - dy_val;

    const float w1 = dx1_val * dy1_val;
    const float w2 = dx_val * dy1_val;
    const float w3 = dx1_val * dy_val;
    const float w4 = dx_val * dy_val;
    return a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *dx, const ITensorInfo *dy,
                          const ITensorInfo *offsets, ITensorInfo *output, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16, DataType::F16, DataType::F32, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(output == input);
    ARM_COMPUTE_RETURN_ERROR_ON(info.sampling_policy != SamplingPolicy::CENTER && info.sampling_policy != SamplingPolicy::TOP_LEFT);
    ARM_COMPUTE_UNUSED(info.constant_border_value);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.use_padding, "Padding is not supported");

    const DataLayout data_layout   = input->data_layout();
    const auto       width_index   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const auto       height_index  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const auto       output_width  = output->dimension(width_index);
    const auto       output_height = output->dimension(height_index);
    ARM_COMPUTE_RETURN_ERROR_ON(output_width == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(output_height == 0);

    if(info.interpolation_policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(offsets, 1, DataType::S32);
    }

    if(info.interpolation_policy == InterpolationPolicy::BILINEAR)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(offsets, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dx, 1, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dy, 1, DataType::F32);
    }

    ARM_COMPUTE_RETURN_ERROR_ON(info.align_corners && !scale_utils::is_align_corners_allowed_sampling_policy(info.sampling_policy));

    if(info.interpolation_policy == InterpolationPolicy::AREA)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(data_layout != DataLayout::NCHW);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    }

    return Status{};
}
} // namespace

NEScaleKernel::NEScaleKernel()
    : _func(nullptr), _offsets(nullptr), _dx(nullptr), _dy(nullptr), _input(nullptr), _output(nullptr), _policy(), _border_mode(), _constant_border_value(PixelValue()), _sampling_offset(0),
      _align_corners(false)
{
}

void NEScaleKernel::configure(const ITensor *input, const ITensor *dx, const ITensor *dy, const ITensor *offsets,
                              ITensor *output, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  dx != nullptr ? dx->info() : nullptr,
                                                  dy != nullptr ? dy->info() : nullptr,
                                                  offsets != nullptr ? offsets->info() : nullptr,
                                                  output->info(),
                                                  info));

    // Get data layout and width/height indices
    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    _input                 = input;
    _output                = output;
    _offsets               = offsets;
    _dx                    = dx;
    _dy                    = dy;
    _policy                = info.interpolation_policy;
    _border_mode           = info.border_mode;
    _constant_border_value = info.constant_border_value;
    _align_corners         = info.align_corners;

    if(info.sampling_policy == SamplingPolicy::CENTER)
    {
        _sampling_offset = 0.5f;
    }

    // Compute the ratio between source width/height and destination width/height
    const auto wr = scale_utils::calculate_resize_ratio(input->info()->dimension(idx_width), output->info()->dimension(idx_width), _align_corners);
    const auto hr = scale_utils::calculate_resize_ratio(input->info()->dimension(idx_height), output->info()->dimension(idx_height), _align_corners);

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    const auto policy_to_use = (info.interpolation_policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f) ? InterpolationPolicy::NEAREST_NEIGHBOR : _policy;

    if(_border_mode == BorderMode::UNDEFINED)
    {
        _border_mode           = BorderMode::CONSTANT;
        _constant_border_value = PixelValue();
    }
    std::string function_to_call("scale_");
    function_to_call += string_from_data_type(_input->info()->data_type()) + "_";
    function_to_call += string_from_data_layout(_input->info()->data_layout()) + "_";
    function_to_call += string_from_interpolation_policy(policy_to_use);

    static std::map<std::string, ScaleFunctionPtr> map_function =
    {
        { "scale_U8_NCHW_AREA_CONSTANT", &NEScaleKernel::scale_area_nchw_u8 },

        { "scale_U8_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_nchw<uint8_t> },
        { "scale_U8_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<uint8_t> },

        { "scale_U8_NHWC_BILINEAR", &NEScaleKernel::scale_bilinear_nhwc<uint8_t> },
        { "scale_U8_NHWC_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nhwc<uint8_t> },

        { "scale_QASYMM8_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_qasymm<uint8_t> },
        { "scale_QASYMM8_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<uint8_t> },

        { "scale_QASYMM8_NHWC_BILINEAR", &NEScaleKernel::scale_bilinear_qasymm<uint8_t> },
        { "scale_QASYMM8_NHWC_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nhwc<uint8_t> },

        { "scale_QASYMM8_SIGNED_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_qasymm<int8_t> },
        { "scale_QASYMM8_SIGNED_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<uint8_t> },

        { "scale_QASYMM8_SIGNED_NHWC_BILINEAR", &NEScaleKernel::scale_bilinear_qasymm<int8_t> },
        { "scale_QASYMM8_SIGNED_NHWC_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nhwc<uint8_t> },

        { "scale_S16_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_nchw<int16_t> },
        { "scale_S16_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<uint16_t> },

        { "scale_S16_NHWC_BILINEAR", &NEScaleKernel::scale_bilinear_nhwc<int16_t> },
        { "scale_S16_NHWC_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nhwc<uint16_t> },

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        { "scale_F16_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_nchw<float16_t> },
        { "scale_F16_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<uint16_t> },

        { "scale_F16_NHWC_BILINEAR", &NEScaleKernel::scale_bilinear_nhwc<float16_t> },
        { "scale_F16_NHWC_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nhwc<uint16_t> },
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

        { "scale_F32_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_nchw<float> },
        { "scale_F32_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<float> },

        { "scale_F32_NHWC_BILINEAR", &NEScaleKernel::scale_bilinear_nhwc<float> },
        { "scale_F32_NHWC_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nhwc<float> },
    };
    auto it = map_function.find(function_to_call);
    if(it != map_function.end())
    {
        _func = it->second;
    }

    // Configure window
    Window      win = calculate_max_window(*output->info(), Steps());
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));
    INEKernel::configure(win);
}

template <typename T>
void NEScaleKernel::scale_nearest_nchw(const Window &window)
{
    const size_t in_dim_x = _input->info()->dimension(0);

    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(_input->info()->dimension(1), _output->info()->dimension(1), _align_corners);

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    // Set offsets window
    Window win_off;
    win_off.set(Window::DimX, window[Window::DimX]);
    win_off.set(Window::DimY, window[Window::DimY]);
    for(size_t d = Window::DimZ; d < _offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    // Create iterators
    Iterator in(_input, win_in);
    Iterator out(_output, window);
    Iterator offsets(_offsets, win_off);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto    offsets_ptr         = reinterpret_cast<const int32_t *>(offsets.ptr());
        const auto    in_yi               = static_cast<int32_t>(_align_corners ? utils::rounding::round_half_away_from_zero((id.y() + _sampling_offset) * hr) : std::floor((id.y() + _sampling_offset) * hr));
        const int32_t offset_row          = in_yi * in_dim_x;
        *reinterpret_cast<T *>(out.ptr()) = *(reinterpret_cast<const T *>(in.ptr()) + offsets_ptr[0] + offset_row);
    },
    in, offsets, out);
}

template <typename T>
void NEScaleKernel::scale_bilinear_nchw(const Window &window)
{
    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(_input->info()->dimension(1), _output->info()->dimension(1), _align_corners);
    Window     win_off;
    win_off.set(Window::DimX, window.x());
    win_off.set(Window::DimY, window.y());

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    for(size_t d = Window::DimZ; d < _offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    Iterator in(_input, win_in);
    Iterator out(_output, window);
    Iterator offsets(_offsets, win_off);
    Iterator dx(_dx, win_off);
    Iterator dy(_dy, win_off);

    const int32_t in_dim_w = _input->info()->dimension(0);
    const int32_t in_dim_h = _input->info()->dimension(1);

    if(_border_mode == BorderMode::CONSTANT)
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        using ConstType = typename std::conditional<std::is_same<T, float16_t>::value, half, T>::type;
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        using ConstType = T;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        const T const_border_value = static_cast<T>(_constant_border_value.get<ConstType>());
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int32_t index_h       = std::floor((id.y() + _sampling_offset) * hr - _sampling_offset);
            const auto    index_w       = *(reinterpret_cast<const int32_t *>(offsets.ptr()));
            const auto    dx_val        = *(reinterpret_cast<const float *>(dx.ptr()));
            const auto    dy_val        = *(reinterpret_cast<const float *>(dy.ptr()));
            const auto    pixel_row_ptr = reinterpret_cast<const T *>(in.ptr());

            const auto a00 = (0 <= index_w && index_w < in_dim_w && 0 <= index_h && index_h < in_dim_h) ? (*(pixel_row_ptr + index_w + index_h * in_dim_w)) : const_border_value;
            const auto a01 = (-1 <= index_w && index_w < in_dim_w - 1 && 0 <= index_h && index_h < in_dim_h) ? (*(pixel_row_ptr + index_w + 1 + index_h * in_dim_w)) : const_border_value;
            const auto a10 = (0 <= index_w && index_w < in_dim_w && -1 <= index_h
                              && index_h < in_dim_h - 1) ?
                             (*(pixel_row_ptr + index_w + index_h * in_dim_w + in_dim_w)) :
                             const_border_value;
            const auto a11 = (-1 <= index_w && index_w < in_dim_w - 1 && -1 <= index_h
                              && index_h < in_dim_h - 1) ?
                             (*(pixel_row_ptr + index_w + 1 + index_h * in_dim_w + in_dim_w)) :
                             const_border_value;

            *reinterpret_cast<T *>(out.ptr()) = static_cast<T>(compute_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, offsets, dx, dy, out);
    }
    else if(_border_mode == BorderMode::REPLICATE)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int  index_h       = std::floor((id.y() + _sampling_offset) * hr - _sampling_offset);
            const auto index_w       = *(reinterpret_cast<const int32_t *>(offsets.ptr()));
            const auto dx_val        = *(reinterpret_cast<const float *>(dx.ptr()));
            const auto dy_val        = *(reinterpret_cast<const float *>(dy.ptr()));
            const auto pixel_row_ptr = reinterpret_cast<const T *>(in.ptr());

            auto clamped_x  = utility::clamp<int>(index_w, 0, in_dim_w - 1);
            auto clamped_x1 = utility::clamp<int>(index_w + 1, 0, in_dim_w - 1);
            auto clamped_y  = utility::clamp<int>(index_h, 0, in_dim_h - 1);
            auto clamped_y1 = utility::clamp<int>(index_h + 1, 0, in_dim_h - 1);

            const auto a00 = *(pixel_row_ptr + clamped_x + clamped_y * in_dim_w);
            const auto a01 = *(pixel_row_ptr + clamped_x1 + clamped_y * in_dim_w);
            const auto a10 = *(pixel_row_ptr + clamped_x + clamped_y1 * in_dim_w);
            const auto a11 = *(pixel_row_ptr + clamped_x1 + clamped_y1 * in_dim_w);

            *reinterpret_cast<T *>(out.ptr()) = static_cast<T>(compute_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, offsets, dx, dy, out);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}

void NEScaleKernel::scale_area_nchw_u8(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(_input, 1, DataType::U8);

    // Don't increment in width/height/channels for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    const auto   wr        = scale_utils::calculate_resize_ratio(_input->info()->dimension(0), _output->info()->dimension(0), _align_corners);
    const auto   hr        = scale_utils::calculate_resize_ratio(_input->info()->dimension(1), _output->info()->dimension(1), _align_corners);
    const auto   w         = _input->info()->dimension(0);
    const auto   h         = _input->info()->dimension(1);
    const size_t in_stride = _input->info()->strides_in_bytes()[1];

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr = reinterpret_cast<const uint8_t *>(in.ptr());

        uint8x8_t tmp0 = vdup_n_u8(0);
        tmp0           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x(), id.y()), tmp0, 0);
        tmp0           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 1, id.y()), tmp0, 1);
        tmp0           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 2, id.y()), tmp0, 2);
        tmp0           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 3, id.y()), tmp0, 3);
        tmp0           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 4, id.y()), tmp0, 4);
        tmp0           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 5, id.y()), tmp0, 5);
        tmp0           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 6, id.y()), tmp0, 6);
        tmp0           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 7, id.y()), tmp0, 7);

        uint8x8_t tmp1 = vdup_n_u8(0);
        tmp1           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 8, id.y()), tmp1, 0);
        tmp1           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 9, id.y()), tmp1, 1);
        tmp1           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 10, id.y()), tmp1, 2);
        tmp1           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 11, id.y()), tmp1, 3);
        tmp1           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 12, id.y()), tmp1, 4);
        tmp1           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 13, id.y()), tmp1, 5);
        tmp1           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 14, id.y()), tmp1, 6);
        tmp1           = vset_lane_u8(pixel_area_c1u8_clamp(in_ptr, in_stride, w, h, wr, hr, id.x() + 15, id.y()), tmp1, 7);

        vst1q_u8(out.ptr(), vcombine_u8(tmp0, tmp1));
    },
    in, out);
}

template <typename T>
void NEScaleKernel::scale_nearest_nhwc(const Window &window)
{
    const size_t in_dim_w  = _input->info()->dimension(1);
    const size_t in_dim_h  = _input->info()->dimension(2);
    const size_t in_dim_c  = _input->info()->dimension(0);
    const size_t in_dim_wc = in_dim_w * in_dim_c;

    // Compute the ratio between source height and destination height
    const auto hr             = scale_utils::calculate_resize_ratio(in_dim_h, _output->info()->dimension(2), _align_corners);
    const auto window_start_x = static_cast<int32_t>(window.x().start());
    const auto window_end_x   = static_cast<int32_t>(window.x().end());
    const int  window_step_x  = 16 / sizeof(T);

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
    Iterator in(_input, win_in);
    Iterator out(_output, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const int32_t offset     = *reinterpret_cast<const int32_t *>(_offsets->ptr_to_element(Coordinates(id.y(), id.z()))) * in_dim_c;
        const auto    in_hi      = static_cast<int>(_align_corners ? utils::rounding::round_half_away_from_zero((id.z() + _sampling_offset) * hr) : std::floor((id.z() + _sampling_offset) * hr));
        const int     offset_row = in_hi * in_dim_wc;
        int32_t       x          = window_start_x;
        for(; x <= window_end_x - window_step_x; x += window_step_x)
        {
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x,
                            wrapper::vloadq(reinterpret_cast<const T *>(in.ptr()) + offset + offset_row + x));
        }
        for(; x < window_end_x; ++x)
        {
            *(reinterpret_cast<T *>(out.ptr()) + x) = *(reinterpret_cast<const T *>(in.ptr()) + offset + offset_row + x);
        }
    },
    in, out);
}

template <typename T>
void NEScaleKernel::scale_bilinear_nhwc(const Window &window)
{
    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(_input->info()->dimension(2), _output->info()->dimension(2), _align_corners);

    Iterator  out(_output, window);
    const int in_dim_c = _input->info()->dimension(0);
    const int in_dim_w = _input->info()->dimension(1);
    const int in_dim_h = _input->info()->dimension(2);
    const int input_wc = in_dim_c * in_dim_w;

    // Don't increment in Y and Z direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
    Iterator in(_input, win_in);

    if(_border_mode == BorderMode::CONSTANT)
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        using ConstType = typename std::conditional<std::is_same<T, float16_t>::value, half, T>::type;
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        using ConstType = T;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        const T const_border_value = static_cast<T>(_constant_border_value.get<ConstType>());
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto    offset = *reinterpret_cast<const int32_t *>(_offsets->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto    dx_val = *reinterpret_cast<const float *>(_dx->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto    dy_val = *reinterpret_cast<const float *>(_dy->ptr_to_element(Coordinates(id.y(), id.z())));
            const int32_t in_hi  = std::floor((id.z() + _sampling_offset) * hr - _sampling_offset);
            const T      *in_ptr = reinterpret_cast<const T *>(in.ptr()) + offset * in_dim_c + in_hi * input_wc;

            const auto a00 = (0 <= offset && offset < in_dim_w && 0 <= in_hi && in_hi < in_dim_h) ? *in_ptr : const_border_value;
            const auto a01 = (-1 <= offset && offset < in_dim_w - 1 && 0 <= in_hi && in_hi < in_dim_h) ? *(in_ptr + in_dim_c) : const_border_value;
            const auto a10 = (0 <= offset && offset < in_dim_w && -1 <= in_hi && in_hi < in_dim_h - 1) ? *(in_ptr + input_wc) : const_border_value;
            const auto a11 = (-1 <= offset && offset < in_dim_w - 1 && -1 <= in_hi && in_hi < in_dim_h - 1) ? *(in_ptr + in_dim_c + input_wc) : const_border_value;

            *reinterpret_cast<T *>(out.ptr()) = static_cast<T>(compute_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, out);
    }
    else if(_border_mode == BorderMode::REPLICATE)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto offset = *reinterpret_cast<const int32_t *>(_offsets->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto dx_val = *reinterpret_cast<const float *>(_dx->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto dy_val = *reinterpret_cast<const float *>(_dy->ptr_to_element(Coordinates(id.y(), id.z())));
            const int  in_hi  = std::floor((id.z() + _sampling_offset) * hr - _sampling_offset);

            auto clamped_w  = utility::clamp<int>(offset, 0, in_dim_w - 1);
            auto clamped_w1 = utility::clamp<int>(offset + 1, 0, in_dim_w - 1);
            auto clamped_h  = utility::clamp<int>(in_hi, 0, in_dim_h - 1);
            auto clamped_h1 = utility::clamp<int>(in_hi + 1, 0, in_dim_h - 1);

            const auto a00 = *(reinterpret_cast<const T *>(in.ptr()) + clamped_w * in_dim_c + clamped_h * input_wc);
            const auto a01 = *(reinterpret_cast<const T *>(in.ptr()) + clamped_w1 * in_dim_c + clamped_h * input_wc);
            const auto a10 = *(reinterpret_cast<const T *>(in.ptr()) + clamped_w * in_dim_c + clamped_h1 * input_wc);
            const auto a11 = *(reinterpret_cast<const T *>(in.ptr()) + clamped_w1 * in_dim_c + clamped_h1 * input_wc);

            *reinterpret_cast<T *>(out.ptr()) = static_cast<T>(compute_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, out);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}

template <typename T>
void NEScaleKernel::scale_bilinear_qasymm(const Window &window)
{
    // Get data layout and width/height indices
    const DataLayout data_layout = _input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(_input->info()->dimension(idx_height), _output->info()->dimension(idx_height), _align_corners);
    Window     win_off;
    win_off.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_off.set(Window::DimY, Window::Dimension(0, 0, 0));

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(idx_width, Window::Dimension(0, 0, 0));
    win_in.set(idx_height, Window::Dimension(0, 0, 0));

    for(size_t d = Window::DimZ; d < _offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    Iterator in(_input, win_in);
    Iterator out(_output, window);

    const int32_t in_dim_w = _input->info()->dimension(idx_width);
    const int32_t in_dim_h = _input->info()->dimension(idx_height);
    const int32_t stride_w = _input->info()->strides_in_bytes()[idx_width];
    const int32_t stride_h = _input->info()->strides_in_bytes()[idx_height];

    const UniformQuantizationInfo iq_info = _input->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info = _output->info()->quantization_info().uniform();

    if(_border_mode == BorderMode::CONSTANT)
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        using ConstType = typename std::conditional<std::is_same<T, float16_t>::value, half, T>::type;
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        using ConstType = T;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        const T const_border_value = static_cast<T>(_constant_border_value.get<ConstType>());
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int32_t index_h       = std::floor((id[idx_height] + _sampling_offset) * hr - _sampling_offset);
            const int32_t index_w       = *(reinterpret_cast<const int32_t *>(_offsets->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dx_val        = *(reinterpret_cast<const float *>(_dx->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dy_val        = *(reinterpret_cast<const float *>(_dy->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    pixel_row_ptr = reinterpret_cast<const T *>(in.ptr());

            const auto a00 = (0 <= index_w && index_w < in_dim_w && 0 <= index_h && index_h < in_dim_h) ?
                             (*(pixel_row_ptr + index_w * stride_w + index_h * stride_h)) :
                             const_border_value;
            const auto a01 = (-1 <= index_w && index_w < in_dim_w - 1 && 0 <= index_h && index_h < in_dim_h) ?
                             (*(pixel_row_ptr + (index_w + 1) * stride_w + index_h * stride_h)) :
                             const_border_value;
            const auto a10 = (0 <= index_w && index_w < in_dim_w && -1 <= index_h && index_h < in_dim_h - 1) ?
                             (*(pixel_row_ptr + index_w * stride_w + (index_h + 1) * stride_h)) :
                             const_border_value;
            const auto a11 = (-1 <= index_w && index_w < in_dim_w - 1 && -1 <= index_h && index_h < in_dim_h - 1) ?
                             (*(pixel_row_ptr + (index_w + 1) * stride_w + (index_h + 1) * stride_h)) :
                             const_border_value;

            const float inp00                 = Qasymm8QuantizationHelper<T>::dequantize(a00, iq_info);
            const float inp01                 = Qasymm8QuantizationHelper<T>::dequantize(a01, iq_info);
            const float inp10                 = Qasymm8QuantizationHelper<T>::dequantize(a10, iq_info);
            const float inp11                 = Qasymm8QuantizationHelper<T>::dequantize(a11, iq_info);
            *reinterpret_cast<T *>(out.ptr()) = Qasymm8QuantizationHelper<T>::quantize(compute_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
        },
        in, out);
    }
    else if(_border_mode == BorderMode::REPLICATE)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int     index_h       = std::floor((id[idx_height] + _sampling_offset) * hr - _sampling_offset);
            const int32_t index_w       = *(reinterpret_cast<const int32_t *>(_offsets->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dx_val        = *(reinterpret_cast<const float *>(_dx->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dy_val        = *(reinterpret_cast<const float *>(_dy->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    pixel_row_ptr = reinterpret_cast<const T *>(in.ptr());

            auto clamped_w  = utility::clamp<int>(index_w, 0, in_dim_w - 1);
            auto clamped_w1 = utility::clamp<int>(index_w + 1, 0, in_dim_w - 1);
            auto clamped_h  = utility::clamp<int>(index_h, 0, in_dim_h - 1);
            auto clamped_h1 = utility::clamp<int>(index_h + 1, 0, in_dim_h - 1);

            const auto a00 = *(pixel_row_ptr + clamped_w * stride_w + clamped_h * stride_h);
            const auto a01 = *(pixel_row_ptr + clamped_w1 * stride_w + clamped_h * stride_h);
            const auto a10 = *(pixel_row_ptr + clamped_w * stride_w + clamped_h1 * stride_h);
            const auto a11 = *(pixel_row_ptr + clamped_w1 * stride_w + clamped_h1 * stride_h);

            const float inp00                 = Qasymm8QuantizationHelper<T>::dequantize(a00, iq_info);
            const float inp01                 = Qasymm8QuantizationHelper<T>::dequantize(a01, iq_info);
            const float inp10                 = Qasymm8QuantizationHelper<T>::dequantize(a10, iq_info);
            const float inp11                 = Qasymm8QuantizationHelper<T>::dequantize(a11, iq_info);
            *reinterpret_cast<T *>(out.ptr()) = Qasymm8QuantizationHelper<T>::quantize(compute_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
        },
        in, out);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}

Status NEScaleKernel::validate(const ITensorInfo *input, const ITensorInfo *dx, const ITensorInfo *dy,
                               const ITensorInfo *offsets, ITensorInfo *output, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, dx, dy, offsets, output, info));
    return Status{};
}

void NEScaleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
} // namespace arm_compute
