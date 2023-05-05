/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#include "src/cpu/kernels/CpuScaleKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/ScaleHelpers.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/scale/neon/list.h"
#include "src/cpu/kernels/scale/sve/list.h"
#include "support/Rounding.h"

#include <arm_neon.h>
#include <map>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuScaleKernel::ScaleKernel> available_kernels =
{
    {
        "sve_fp16_scale",
        [](const ScaleKernelDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F16 && data.isa.sve && data.isa.fp16 && data.interpolation_policy != InterpolationPolicy::BILINEAR;
        },
        REGISTER_FP16_SVE(arm_compute::cpu::fp16_sve_scale)
    },
    {
        "sve_fp32_scale",
        [](const ScaleKernelDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F32 && data.isa.sve && data.interpolation_policy != InterpolationPolicy::BILINEAR;
        },
        REGISTER_FP32_SVE(arm_compute::cpu::fp32_sve_scale)
    },
    {
        "sve_qu8_scale",
        [](const ScaleKernelDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8 && data.isa.sve && data.interpolation_policy != InterpolationPolicy::BILINEAR;
        },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::qasymm8_sve_scale)
    },
    {
        "sve_qs8_scale",
        [](const ScaleKernelDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8_SIGNED && data.isa.sve && data.interpolation_policy != InterpolationPolicy::BILINEAR;
        },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::qasymm8_signed_sve_scale)
    },
    {
        "sve_u8_scale",
        [](const ScaleKernelDataTypeISASelectorData & data)
        {
            return data.dt == DataType::U8 && data.isa.sve && data.interpolation_policy != InterpolationPolicy::BILINEAR;
        },
        REGISTER_INTEGER_SVE(arm_compute::cpu::u8_sve_scale)
    },
    {
        "sve_s16_scale",
        [](const ScaleKernelDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S16 && data.isa.sve && data.interpolation_policy != InterpolationPolicy::BILINEAR;
        },
        REGISTER_INTEGER_SVE(arm_compute::cpu::s16_sve_scale)
    },
    {
        "neon_fp16_scale",
        [](const ScaleKernelDataTypeISASelectorData & data) { return data.dt == DataType::F16 && data.isa.fp16; },
        REGISTER_FP16_NEON(arm_compute::cpu::common_neon_scale<float16_t>)
    },
    {
        "neon_fp32_scale",
        [](const ScaleKernelDataTypeISASelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::common_neon_scale<float>)
    },
    {
        "neon_qu8_scale",
        [](const ScaleKernelDataTypeISASelectorData & data) { return data.dt == DataType::QASYMM8; },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::qasymm8_neon_scale)
    },
    {
        "neon_qs8_scale",
        [](const ScaleKernelDataTypeISASelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED; },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::qasymm8_signed_neon_scale)
    },
    {
        "neon_u8_scale",
        [](const ScaleKernelDataTypeISASelectorData & data) { return data.dt == DataType::U8; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::u8_neon_scale)
    },
    {
        "neon_s8_scale",
        [](const ScaleKernelDataTypeISASelectorData & data) { return data.dt == DataType::S8; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::s8_neon_scale)
    },
    {
        "neon_s16_scale",
        [](const ScaleKernelDataTypeISASelectorData & data) { return data.dt == DataType::S16; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::s16_neon_scale)
    },
};

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dx, const ITensorInfo *dy,
                          const ITensorInfo *offsets, ITensorInfo *dst, const ScaleKernelInfo &info)
{
    const auto *uk = CpuScaleKernel::get_implementation(ScaleKernelDataTypeISASelectorData{ src->data_type(), CPUInfo::get().get_isa(), info.interpolation_policy });

    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(dst == src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->num_channels()!=1);
    ARM_COMPUTE_RETURN_ERROR_ON(info.sampling_policy != SamplingPolicy::CENTER && info.sampling_policy != SamplingPolicy::TOP_LEFT);
    ARM_COMPUTE_UNUSED(info.constant_border_value);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.use_padding, "Padding is not supported");

    const DataLayout data_layout   = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;
    const auto       width_index   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const auto       height_index  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const auto       output_width  = dst->dimension(width_index);
    const auto       output_height = dst->dimension(height_index);
    ARM_COMPUTE_RETURN_ERROR_ON(output_width == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(output_height == 0);

    ARM_COMPUTE_RETURN_ERROR_ON((src->data_type() == DataType::S8) && (data_layout != DataLayout::NHWC || info.interpolation_policy != InterpolationPolicy::BILINEAR
                                                                       || info.border_mode != BorderMode::REPLICATE));

    if(info.interpolation_policy == InterpolationPolicy::NEAREST_NEIGHBOR && offsets != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(offsets, 1, DataType::S32);
    }

    if(info.interpolation_policy == InterpolationPolicy::BILINEAR && offsets != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(offsets, 1, DataType::S32);
        if(dx != nullptr && dy != nullptr)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dx, 1, DataType::F32);
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dy, 1, DataType::F32);
        }
    }

    ARM_COMPUTE_RETURN_ERROR_ON(info.align_corners && !scale_utils::is_align_corners_allowed_sampling_policy(info.sampling_policy));

    if(info.interpolation_policy == InterpolationPolicy::AREA)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(data_layout != DataLayout::NCHW);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::U8);
    }

    return Status{};
}
} // namespace

void CpuScaleKernel::configure(const ITensorInfo *src, const ITensorInfo *dx, const ITensorInfo *dy, const ITensorInfo *offsets,
                               ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_UNUSED(dx, dy, offsets);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src,
                                                  dx,
                                                  dy,
                                                  offsets,
                                                  dst,
                                                  info));

    const auto *uk = CpuScaleKernel::get_implementation(ScaleKernelDataTypeISASelectorData{ src->data_type(), CPUInfo::get().get_isa(), info.interpolation_policy });
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _run_method = uk->ukernel;
    _name       = std::string("CpuScaleKernel").append("/").append(uk->name).append("_").append(string_from_interpolation_policy(info.interpolation_policy));

    // Get data layout and width/height indices
    _data_layout         = info.data_layout == DataLayout::UNKNOWN ? src->data_layout() : info.data_layout;
    const int idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

    _policy                = info.interpolation_policy;
    _border_mode           = info.border_mode;
    _constant_border_value = info.constant_border_value;
    _align_corners         = info.align_corners;

    if(info.sampling_policy == SamplingPolicy::CENTER)
    {
        _sampling_offset = 0.5f;
    }

    // Compute the ratio between source width/height and destination width/height
    const auto wr = scale_utils::calculate_resize_ratio(src->dimension(idx_width), dst->dimension(idx_width), _align_corners);
    const auto hr = scale_utils::calculate_resize_ratio(src->dimension(idx_height), dst->dimension(idx_height), _align_corners);

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    _policy = (_policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f) ? InterpolationPolicy::NEAREST_NEIGHBOR : _policy;

    if(_border_mode == BorderMode::UNDEFINED)
    {
        _border_mode           = BorderMode::CONSTANT;
        _constant_border_value = PixelValue();
    }

#ifdef ENABLE_NCHW_KERNELS
    // Configure scale function to run
    if(_data_layout == DataLayout::NCHW)
    {
        std::string function_to_call("scale_");
        function_to_call += string_from_data_type(src->data_type()) + "_";
        function_to_call += string_from_data_layout(_data_layout) + "_";
        function_to_call += string_from_interpolation_policy(_policy);

        static std::map<std::string, ScaleFunctionPtr> map_function =
        {
            { "scale_U8_NCHW_AREA_CONSTANT", &CpuScaleKernel::scale_area_nchw_u8 },

            { "scale_U8_NCHW_BILINEAR", &CpuScaleKernel::scale_bilinear_nchw<uint8_t> },
            { "scale_U8_NCHW_NEAREST_NEIGHBOUR", &CpuScaleKernel::scale_nearest_nchw<uint8_t> },

            { "scale_QASYMM8_NCHW_BILINEAR", &CpuScaleKernel::scale_bilinear_qasymm<uint8_t> },
            { "scale_QASYMM8_NCHW_NEAREST_NEIGHBOUR", &CpuScaleKernel::scale_nearest_nchw<uint8_t> },

            { "scale_QASYMM8_SIGNED_NCHW_BILINEAR", &CpuScaleKernel::scale_bilinear_qasymm<int8_t> },
            { "scale_QASYMM8_SIGNED_NCHW_NEAREST_NEIGHBOUR", &CpuScaleKernel::scale_nearest_nchw<int8_t> },

            { "scale_S16_NCHW_BILINEAR", &CpuScaleKernel::scale_bilinear_nchw<int16_t> },
            { "scale_S16_NCHW_NEAREST_NEIGHBOUR", &CpuScaleKernel::scale_nearest_nchw<int16_t> },

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            { "scale_F16_NCHW_BILINEAR", &CpuScaleKernel::scale_bilinear_nchw<float16_t> },
            { "scale_F16_NCHW_NEAREST_NEIGHBOUR", &CpuScaleKernel::scale_nearest_nchw<float16_t> },
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

            { "scale_F32_NCHW_BILINEAR", &CpuScaleKernel::scale_bilinear_nchw<float> },
            { "scale_F32_NCHW_NEAREST_NEIGHBOUR", &CpuScaleKernel::scale_nearest_nchw<float> },
        };
        auto it = map_function.find(function_to_call);
        if(it != map_function.end())
        {
            _func = it->second;
        }
    }
#endif // ENABLE_NCHW_KERNELS

    // Configure window
    Window win = calculate_max_window(*dst, Steps());
    ICpuKernel::configure(win);
}

#ifdef ENABLE_NCHW_KERNELS
template <typename T>
void CpuScaleKernel::scale_nearest_nchw(const ITensor *src, ITensor *dst, const ITensor *dx, const ITensor *dy, const ITensor *offsets, const Window &window)
{
    ARM_COMPUTE_UNUSED(dx, dy);
    const size_t in_stride_x = src->info()->dimension(0) + src->info()->padding().left + src->info()->padding().right;

    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(src->info()->dimension(1), dst->info()->dimension(1), _align_corners);

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    // Set offsets window
    Window win_off;
    win_off.set(Window::DimX, window[Window::DimX]);
    win_off.set(Window::DimY, window[Window::DimY]);
    for(size_t d = Window::DimZ; d < offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    // Create iterators
    Iterator src_i(src, win_in);
    Iterator dst_i(dst, window);
    Iterator offsets_i(offsets, win_off);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets_i.ptr());
        const auto in_yi       = static_cast<int32_t>(_align_corners ? utils::rounding::round_half_away_from_zero((id.y() + _sampling_offset) * hr) : std::floor((
                                                          id.y() + _sampling_offset)
                                                      * hr));
        const int32_t offset_row            = in_yi * in_stride_x;
        *reinterpret_cast<T *>(dst_i.ptr()) = *(reinterpret_cast<const T *>(src_i.ptr()) + offsets_ptr[0] + offset_row);
    },
    src_i, offsets_i, dst_i);
}

template <typename T>
void CpuScaleKernel::scale_bilinear_nchw(const ITensor *src, ITensor *dst, const ITensor *dx, const ITensor *dy, const ITensor *offsets, const Window &window)
{
    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(src->info()->dimension(1), dst->info()->dimension(1), _align_corners);
    Window     win_off;
    win_off.set(Window::DimX, window.x());
    win_off.set(Window::DimY, window.y());

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    for(size_t d = Window::DimZ; d < offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    Iterator src_i(src, win_in);
    Iterator dst_i(dst, window);
    Iterator offsets_i(offsets, win_off);
    Iterator dx_i(dx, win_off);
    Iterator dy_i(dy, win_off);

    const int32_t in_dim_w    = src->info()->dimension(0);
    const int32_t in_dim_h    = src->info()->dimension(1);
    const int32_t in_stride_w = in_dim_w + src->info()->padding().left + src->info()->padding().right;

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
            const auto    index_w       = *(reinterpret_cast<const int32_t *>(offsets_i.ptr()));
            const auto    dx_val        = *(reinterpret_cast<const float *>(dx_i.ptr()));
            const auto    dy_val        = *(reinterpret_cast<const float *>(dy_i.ptr()));
            const auto    pixel_row_ptr = reinterpret_cast<const T *>(src_i.ptr());

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

            *reinterpret_cast<T *>(dst_i.ptr()) = static_cast<T>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        src_i, offsets_i, dx_i, dy_i, dst_i);
    }
    else if(_border_mode == BorderMode::REPLICATE)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int  index_h       = std::floor((id.y() + _sampling_offset) * hr - _sampling_offset);
            const auto index_w       = *(reinterpret_cast<const int32_t *>(offsets_i.ptr()));
            const auto dx_val        = *(reinterpret_cast<const float *>(dx_i.ptr()));
            const auto dy_val        = *(reinterpret_cast<const float *>(dy_i.ptr()));
            const auto pixel_row_ptr = reinterpret_cast<const T *>(src_i.ptr());

            auto clamped_x  = utility::clamp<int>(index_w, 0, in_dim_w - 1);
            auto clamped_x1 = utility::clamp<int>(index_w + 1, 0, in_dim_w - 1);
            auto clamped_y  = utility::clamp<int>(index_h, 0, in_dim_h - 1);
            auto clamped_y1 = utility::clamp<int>(index_h + 1, 0, in_dim_h - 1);

            const auto a00 = *(pixel_row_ptr + clamped_x + clamped_y * in_stride_w);
            const auto a01 = *(pixel_row_ptr + clamped_x1 + clamped_y * in_stride_w);
            const auto a10 = *(pixel_row_ptr + clamped_x + clamped_y1 * in_stride_w);
            const auto a11 = *(pixel_row_ptr + clamped_x1 + clamped_y1 * in_stride_w);

            *reinterpret_cast<T *>(dst_i.ptr()) = static_cast<T>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        src_i, offsets_i, dx_i, dy_i, dst_i);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}

void CpuScaleKernel::scale_area_nchw_u8(const ITensor *src, ITensor *dst, const ITensor *dx, const ITensor *dy, const ITensor *offsets, const Window &window)
{
    ARM_COMPUTE_UNUSED(dx, dy, offsets);
    using namespace scale_helpers;

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::U8);

    // Don't increment in width/height/channels for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Iterator src_i(src, win_in);
    Iterator dst_i(dst, window);

    const auto   wr        = scale_utils::calculate_resize_ratio(src->info()->dimension(0), dst->info()->dimension(0), _align_corners);
    const auto   hr        = scale_utils::calculate_resize_ratio(src->info()->dimension(1), dst->info()->dimension(1), _align_corners);
    const auto   w         = src->info()->dimension(0);
    const auto   h         = src->info()->dimension(1);
    const size_t in_stride = src->info()->strides_in_bytes()[1];

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr = reinterpret_cast<const uint8_t *>(src_i.ptr());

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

        vst1q_u8(dst_i.ptr(), vcombine_u8(tmp0, tmp1));
    },
    src_i, dst_i);
}

template <typename T>
void CpuScaleKernel::scale_bilinear_qasymm(const ITensor *src, ITensor *dst, const ITensor *dx, const ITensor *dy, const ITensor *offsets, const Window &window)
{
    // Get data layout and width/height indices
    const int idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(src->info()->dimension(idx_height), dst->info()->dimension(idx_height), _align_corners);
    Window     win_off;
    win_off.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_off.set(Window::DimY, Window::Dimension(0, 0, 0));

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(idx_width, Window::Dimension(0, 0, 0));
    win_in.set(idx_height, Window::Dimension(0, 0, 0));

    for(size_t d = Window::DimZ; d < offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    Iterator src_i(src, win_in);
    Iterator dst_i(dst, window);

    const int32_t in_dim_w = src->info()->dimension(idx_width);
    const int32_t in_dim_h = src->info()->dimension(idx_height);
    const int32_t stride_w = src->info()->strides_in_bytes()[idx_width];
    const int32_t stride_h = src->info()->strides_in_bytes()[idx_height];

    const UniformQuantizationInfo iq_info = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info = dst->info()->quantization_info().uniform();

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
            const int32_t index_w       = *(reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dx_val        = *(reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dy_val        = *(reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    pixel_row_ptr = reinterpret_cast<const T *>(src_i.ptr());

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

            const float inp00                   = Qasymm8QuantizationHelper<T>::dequantize(a00, iq_info);
            const float inp01                   = Qasymm8QuantizationHelper<T>::dequantize(a01, iq_info);
            const float inp10                   = Qasymm8QuantizationHelper<T>::dequantize(a10, iq_info);
            const float inp11                   = Qasymm8QuantizationHelper<T>::dequantize(a11, iq_info);
            *reinterpret_cast<T *>(dst_i.ptr()) = Qasymm8QuantizationHelper<T>::quantize(scale_helpers::delta_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
        },
        src_i, dst_i);
    }
    else if(_border_mode == BorderMode::REPLICATE)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int     index_h       = std::floor((id[idx_height] + _sampling_offset) * hr - _sampling_offset);
            const int32_t index_w       = *(reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dx_val        = *(reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dy_val        = *(reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    pixel_row_ptr = reinterpret_cast<const T *>(src_i.ptr());

            auto clamped_w  = utility::clamp<int>(index_w, 0, in_dim_w - 1);
            auto clamped_w1 = utility::clamp<int>(index_w + 1, 0, in_dim_w - 1);
            auto clamped_h  = utility::clamp<int>(index_h, 0, in_dim_h - 1);
            auto clamped_h1 = utility::clamp<int>(index_h + 1, 0, in_dim_h - 1);

            const auto a00 = *(pixel_row_ptr + clamped_w * stride_w + clamped_h * stride_h);
            const auto a01 = *(pixel_row_ptr + clamped_w1 * stride_w + clamped_h * stride_h);
            const auto a10 = *(pixel_row_ptr + clamped_w * stride_w + clamped_h1 * stride_h);
            const auto a11 = *(pixel_row_ptr + clamped_w1 * stride_w + clamped_h1 * stride_h);

            const float inp00                   = Qasymm8QuantizationHelper<T>::dequantize(a00, iq_info);
            const float inp01                   = Qasymm8QuantizationHelper<T>::dequantize(a01, iq_info);
            const float inp10                   = Qasymm8QuantizationHelper<T>::dequantize(a10, iq_info);
            const float inp11                   = Qasymm8QuantizationHelper<T>::dequantize(a11, iq_info);
            *reinterpret_cast<T *>(dst_i.ptr()) = Qasymm8QuantizationHelper<T>::quantize(scale_helpers::delta_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
        },
        src_i, dst_i);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}
#endif // ENABLE_NCHW_KERNELS

Status CpuScaleKernel::validate(const ITensorInfo *input, const ITensorInfo *dx, const ITensorInfo *dy,
                                const ITensorInfo *offsets, ITensorInfo *output, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, dx, dy, offsets, output, info));
    return Status{};
}

void CpuScaleKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr && _data_layout == DataLayout::NCHW);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr && _data_layout == DataLayout::NHWC);

    const auto src     = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst     = tensors.get_tensor(TensorType::ACL_DST);
    const auto dx      = tensors.get_const_tensor(TensorType::ACL_INT_0);
    const auto dy      = tensors.get_const_tensor(TensorType::ACL_INT_1);
    const auto offsets = tensors.get_const_tensor(TensorType::ACL_INT_2);

    if(_data_layout == DataLayout::NCHW)
    {
        (this->*_func)(src, dst, dx, dy, offsets, window);
    }
    else
    {
        _run_method(src, dst, offsets, dx, dy, _policy, _border_mode, _constant_border_value, _sampling_offset, _align_corners, window);
    }
}

const char *CpuScaleKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuScaleKernel::ScaleKernel> &CpuScaleKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
