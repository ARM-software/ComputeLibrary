/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEScaleKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/scale/impl/NEON/list.h"
#include "src/core/NEON/kernels/scale/impl/SVE/list.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/ScaleHelpers.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/ScaleUtils.h"
#include "support/Rounding.h"
#include <arm_neon.h>
#include <map>

namespace arm_compute
{
namespace
{
struct ScaleSelectorData
{
    DataType dt;
};
using ScaleSelectorPtr = std::add_pointer<bool(const ScaleSelectorData &data)>::type;
using ScaleKernelPtr   = std::add_pointer<void(const ITensor *, ITensor *, const ITensor *, const ITensor *, const ITensor *,
                                               InterpolationPolicy, BorderMode, PixelValue, float, bool, const Window &)>::type;
struct ScaleKernel
{
    const char            *name;
    const ScaleSelectorPtr is_selected;
    ScaleKernelPtr         ukernel;
};

static const ScaleKernel available_kernels[] =
{
#if defined(__ARM_FEATURE_SVE)
    {
        "fp16_sve_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_NEON(arm_compute::cpu::fp16_sve_scale)
    },
    {
        "f32_sve_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::fp32_sve_scale)
    },
    {
        "qasymm8_sve_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::QASYMM8; },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::qasymm8_sve_scale)
    },
    {
        "qasymm8_signed_sve_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED; },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::qasymm8_signed_sve_scale)
    },
    {
        "u8_sve_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::U8; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::u8_sve_scale)
    },
    {
        "s16_sve_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::S16; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::s16_sve_scale)
    },
#else /* !defined(__ARM_FEATURE_SVE) */
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "common_neon_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_NEON(arm_compute::cpu::common_neon_scale<float16_t>)
    },
#endif /* !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "common_neon_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::common_neon_scale<float>)
    },
    {
        "qasymm8_neon_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::QASYMM8; },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::qasymm8_neon_scale)
    },
    {
        "qasymm8_signed_neon_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED; },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::qasymm8_signed_neon_scale)
    },
    {
        "common_neon_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::U8; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::common_neon_scale<uint8_t>)
    },
    {
        "common_neon_scale",
        [](const ScaleSelectorData & data) { return data.dt == DataType::S16; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::common_neon_scale<int16_t>)
    },
#endif /* !defined(__ARM_FEATURE_SVE) */
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const ScaleKernel *get_implementation(const ScaleSelectorData &data)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected(data))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *dx, const ITensorInfo *dy,
                          const ITensorInfo *offsets, ITensorInfo *output, const ScaleKernelInfo &info)
{
    const auto *uk = get_implementation(ScaleSelectorData{ input->data_type() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(output == input);
    ARM_COMPUTE_RETURN_ERROR_ON(info.sampling_policy != SamplingPolicy::CENTER && info.sampling_policy != SamplingPolicy::TOP_LEFT);
    ARM_COMPUTE_UNUSED(info.constant_border_value);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.use_padding, "Padding is not supported");

    const DataLayout data_layout   = info.data_layout == DataLayout::UNKNOWN ? input->data_layout() : info.data_layout;
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
      _align_corners(false), _data_layout(DataLayout::UNKNOWN)
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
    _data_layout         = info.data_layout == DataLayout::UNKNOWN ? input->info()->data_layout() : info.data_layout;
    const int idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

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
    _policy = (_policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f) ? InterpolationPolicy::NEAREST_NEIGHBOR : _policy;

    if(_border_mode == BorderMode::UNDEFINED)
    {
        _border_mode           = BorderMode::CONSTANT;
        _constant_border_value = PixelValue();
    }

    // Configure scale function to run
    if(_data_layout == DataLayout::NCHW)
    {
        std::string function_to_call("scale_");
        function_to_call += string_from_data_type(_input->info()->data_type()) + "_";
        function_to_call += string_from_data_layout(_data_layout) + "_";
        function_to_call += string_from_interpolation_policy(_policy);

        static std::map<std::string, ScaleFunctionPtr> map_function =
        {
            { "scale_U8_NCHW_AREA_CONSTANT", &NEScaleKernel::scale_area_nchw_u8 },

            { "scale_U8_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_nchw<uint8_t> },
            { "scale_U8_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<uint8_t> },

            { "scale_QASYMM8_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_qasymm<uint8_t> },
            { "scale_QASYMM8_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<uint8_t> },

            { "scale_QASYMM8_SIGNED_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_qasymm<int8_t> },
            { "scale_QASYMM8_SIGNED_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<int8_t> },

            { "scale_S16_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_nchw<int16_t> },
            { "scale_S16_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<int16_t> },

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            { "scale_F16_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_nchw<float16_t> },
            { "scale_F16_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<float16_t> },
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

            { "scale_F32_NCHW_BILINEAR", &NEScaleKernel::scale_bilinear_nchw<float> },
            { "scale_F32_NCHW_NEAREST_NEIGHBOUR", &NEScaleKernel::scale_nearest_nchw<float> },
        };
        auto it = map_function.find(function_to_call);
        if(it != map_function.end())
        {
            _func = it->second;
        }
    }

    // Configure window
    Window win = calculate_max_window(*output->info(), Steps());
    INEKernel::configure(win);
}

template <typename T>
void NEScaleKernel::scale_nearest_nchw(const Window &window)
{
    const size_t in_stride_x = _input->info()->dimension(0) + _input->info()->padding().left + _input->info()->padding().right;

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
        const int32_t offset_row          = in_yi * in_stride_x;
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

    const int32_t in_dim_w    = _input->info()->dimension(0);
    const int32_t in_dim_h    = _input->info()->dimension(1);
    const int32_t in_stride_w = in_dim_w + _input->info()->padding().left + _input->info()->padding().right;

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

            const auto a00 = (0 <= index_w && index_w < in_dim_w && 0 <= index_h && index_h < in_dim_h) ? (*(pixel_row_ptr + index_w + index_h * in_stride_w)) : const_border_value;
            const auto a01 = (-1 <= index_w && index_w < in_dim_w - 1 && 0 <= index_h && index_h < in_dim_h) ? (*(pixel_row_ptr + index_w + 1 + index_h * in_stride_w)) : const_border_value;
            const auto a10 = (0 <= index_w && index_w < in_dim_w && -1 <= index_h
                              && index_h < in_dim_h - 1) ?
                             (*(pixel_row_ptr + index_w + index_h * in_stride_w + in_stride_w)) :
                             const_border_value;
            const auto a11 = (-1 <= index_w && index_w < in_dim_w - 1 && -1 <= index_h
                              && index_h < in_dim_h - 1) ?
                             (*(pixel_row_ptr + index_w + 1 + index_h * in_stride_w + in_stride_w)) :
                             const_border_value;

            *reinterpret_cast<T *>(out.ptr()) = static_cast<T>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
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

            const auto a00 = *(pixel_row_ptr + clamped_x + clamped_y * in_stride_w);
            const auto a01 = *(pixel_row_ptr + clamped_x1 + clamped_y * in_stride_w);
            const auto a10 = *(pixel_row_ptr + clamped_x + clamped_y1 * in_stride_w);
            const auto a11 = *(pixel_row_ptr + clamped_x1 + clamped_y1 * in_stride_w);

            *reinterpret_cast<T *>(out.ptr()) = static_cast<T>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
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
    using namespace scale_helpers;

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
void NEScaleKernel::scale_bilinear_qasymm(const Window &window)
{
    // Get data layout and width/height indices
    const int idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

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
            *reinterpret_cast<T *>(out.ptr()) = Qasymm8QuantizationHelper<T>::quantize(scale_helpers::delta_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
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
            *reinterpret_cast<T *>(out.ptr()) = Qasymm8QuantizationHelper<T>::quantize(scale_helpers::delta_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
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
    ARM_COMPUTE_ERROR_ON(_func == nullptr && _data_layout == DataLayout::NCHW);

    if(_data_layout == DataLayout::NCHW)
    {
        (this->*_func)(window);
    }
    else
    {
        const auto *uk = get_implementation(ScaleSelectorData{ _input->info()->data_type() });
        uk->ukernel(_input, _output, _offsets, _dx, _dy, _policy, _border_mode, _constant_border_value, _sampling_offset, _align_corners, window);
    }
}
} // namespace arm_compute
