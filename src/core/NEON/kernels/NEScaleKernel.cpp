/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/Utility.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *dx, const ITensorInfo *dy,
                          const ITensorInfo *offsets, ITensorInfo *output, InterpolationPolicy policy,
                          BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy, bool use_padding)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16, DataType::F16, DataType::F32, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(output == input);
    ARM_COMPUTE_RETURN_ERROR_ON(sampling_policy != SamplingPolicy::CENTER && sampling_policy != SamplingPolicy::TOP_LEFT);
    ARM_COMPUTE_RETURN_ERROR_ON(!use_padding && border_mode != BorderMode::CONSTANT);
    ARM_COMPUTE_UNUSED(constant_border_value);

    const DataLayout data_layout = input->data_layout();
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH)) == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT)) == 0);

    if(policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(offsets, 1, DataType::S32);
    }

    if(policy == InterpolationPolicy::BILINEAR)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(offsets, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dx, 1, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dy, 1, DataType::F32);
    }

    if(policy == InterpolationPolicy::AREA)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(data_layout != DataLayout::NCHW);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_nchw(ITensorInfo *input, ITensorInfo *dx, ITensorInfo *dy, ITensorInfo *offsets, ITensorInfo *output,
                                                             InterpolationPolicy policy, bool border_undefined, SamplingPolicy sampling_policy, BorderSize border_size)
{
    bool   window_changed{ false };
    Window win{};

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    const ValidRegion &input_valid_region = input->valid_region();

    if(offsets != nullptr)
    {
        AccessWindowHorizontal offsets_access(offsets, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, offsets_access);
    }
    if(dx != nullptr && dy != nullptr)
    {
        AccessWindowHorizontal dx_access(dx, 0, num_elems_processed_per_iteration);
        AccessWindowHorizontal dy_access(dy, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, dx_access, dy_access);
    }

    // Reads can occur within the valid region of the input
    AccessWindowStatic input_access(input, input_valid_region.anchor[0] - border_size.left,
                                    input_valid_region.anchor[1] - border_size.top,
                                    input_valid_region.anchor[0] + input_valid_region.shape[0] + border_size.right,
                                    input_valid_region.anchor[1] + input_valid_region.shape[1] + border_size.bottom);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    window_changed = window_changed || update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, calculate_valid_region_scale(*input, output->tensor_shape(),
                                                                     policy, sampling_policy, border_undefined));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window_nhwc(ITensorInfo *input, ITensorInfo *output,
                                                             InterpolationPolicy policy, bool border_undefined,
                                                             SamplingPolicy sampling_policy, BorderSize border_size, bool use_padding)
{
    bool   window_changed{ false };
    Window win{};

    const unsigned int num_elems_processed_per_iteration = (use_padding && policy == InterpolationPolicy::NEAREST_NEIGHBOR) ? 16 / input->element_size() : 1;

    // Configure kernel window
    win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    if(use_padding)
    {
        AccessWindowStatic input_access(input, 0, -border_size.top, use_padding ? ceil_to_multiple(input->tensor_shape()[0], num_elems_processed_per_iteration) : num_elems_processed_per_iteration,
                                        input->tensor_shape()[1]);
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output->set_valid_region(calculate_valid_region_scale(*input, output->tensor_shape(), policy, sampling_policy, border_undefined));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *dx, ITensorInfo *dy, ITensorInfo *offsets, ITensorInfo *output,
                                                        InterpolationPolicy policy, bool border_undefined, SamplingPolicy sampling_policy, BorderSize border_size, bool use_padding)
{
    std::pair<Status, Window> win_config;
    switch(input->data_layout())
    {
        case DataLayout::NCHW:
            if(!use_padding)
            {
                return std::make_pair(ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Padding required for NCHW"), Window{});
            }
            win_config = validate_and_configure_window_nchw(input, dx, dy, offsets, output, policy, border_undefined, sampling_policy, border_size);
            break;
        case DataLayout::NHWC:
            win_config = validate_and_configure_window_nhwc(input, output, policy, border_undefined, sampling_policy, border_size, use_padding);
            break;
        default:
            win_config = std::make_pair(ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported data layout!"), Window{});
    }

    return win_config;
}

template <typename T>
inline void scale_nearest_nhwc_core(const ITensor *input, const ITensor *offsets, ITensor *output,
                                    float hr, Window window, const Window &win_in, size_t stride_w, size_t stride_h, size_t stride_c)
{
    const int  window_step_x  = 16 / sizeof(T);
    const auto window_start_x = static_cast<int32_t>(window.x().start());
    const auto window_end_x   = static_cast<int32_t>(window.x().end());

    window.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(input, win_in);
    Iterator out(output, window);

    const size_t offsets_stride = stride_w / sizeof(T);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int32_t offset     = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z())));
        const int     in_yi      = (id.z() + 0.5f) * hr;
        const int     offset_row = in_yi * stride_h;
        int32_t       x          = window_start_x;
        for(; x < window_end_x - window_step_x; x += window_step_x)
        {
            wrapper::vstore(reinterpret_cast<T *>(out.ptr()) + x,
                            wrapper::vloadq(reinterpret_cast<const T *>(in.ptr() + offset * offsets_stride + offset_row + x * stride_c)));
        }
        for(; x < window_end_x; ++x)
        {
            *(reinterpret_cast<T *>(out.ptr()) + x) =
                *(reinterpret_cast<const T *>(in.ptr() + offset * offsets_stride + offset_row + x * stride_c));
        }
    },
    in, out);
}

template <typename T, typename ConstType>
inline void scale_bilinear_nhwc_core(const ITensor *input, const ITensor *offsets, const ITensor *dx, const ITensor *dy, ITensor *output,
                                     float hr, float sampling_offset, Window window, const Window &win_in, size_t stride_w, size_t stride_h,
                                     size_t stride_c, BorderMode border_mode, PixelValue constant_border_value, bool use_padding)
{
    Iterator in(input, win_in);
    Iterator out(output, window);

    const size_t stride_w_elems = stride_w / sizeof(T);
    const size_t stride_h_elems = stride_h / sizeof(T);

    const int input_width  = input->info()->dimension(1);
    const int input_height = input->info()->dimension(2);

    T border_value;
    if(use_padding)
    {
        border_value = *reinterpret_cast<T *>(input->buffer() + input->info()->offset_first_element_in_bytes() - stride_w);
    }
    else
    {
        border_value = static_cast<T>(constant_border_value.get<ConstType>());
    }

    auto is_valid = [](int x, int low_x, int high_x, int y, int low_y, int high_y)
    {
        return !(x < low_x || x > high_x || y < low_y || y > high_y);
    };

    int border_size = (border_mode == BorderMode::UNDEFINED) ? 0 : 1;

    const bool             is_quantized = (input->info()->data_type() == DataType::QASYMM8);
    const QuantizationInfo iq_info      = input->info()->quantization_info();
    const QuantizationInfo oq_info      = output->info()->quantization_info();

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto offset     = (*reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z())))) / static_cast<int>(sizeof(T));
        const auto dx_scale   = *reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id.y(), id.z())));
        const auto dy_scale   = *reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id.y(), id.z())));
        const int  in_yi      = std::floor((id.z() + sampling_offset) * hr - sampling_offset);
        const int  offset_row = in_yi * stride_h + id.x() * stride_c;
        const T   *in_ptr     = reinterpret_cast<T *>(in.ptr() + offset * stride_w + offset_row);

        if(is_valid(offset, -border_size, input_width - 1 + border_size, in_yi, -border_size, input_height - 1 + border_size))
        {
            T a00 = 0;
            T a01 = 0;
            T a10 = 0;
            T a11 = 0;

            if(border_mode == BorderMode::CONSTANT)
            {
                a00 = is_valid(offset, 0, input_width - 1, in_yi, 0, input_height - 1) ? *in_ptr : border_value;
                a01 = is_valid(offset + 1, 0, input_width - 1, in_yi, 0, input_height - 1) ? *(in_ptr + stride_w_elems) : border_value;
                a10 = is_valid(offset, 0, input_width - 1, in_yi + 1, 0, input_height - 1) ? *(in_ptr + stride_h_elems) : border_value;
                a11 = is_valid(offset + 1, 0, input_width - 1, in_yi + 1, 0, input_height - 1) ? *(in_ptr + stride_h_elems + stride_w_elems) : border_value;
            }
            else if(border_mode == BorderMode::REPLICATE)
            {
                auto clamped_x  = utility::clamp<int>(offset, 0, input_width - 1);
                auto clamped_x1 = utility::clamp<int>(offset + 1, 0, input_width - 1);
                auto clamped_y  = utility::clamp<int>(in_yi, 0, input_height - 1);
                auto clamped_y1 = utility::clamp<int>(in_yi + 1, 0, input_height - 1);

                a00 = *reinterpret_cast<T *>(in.ptr() + clamped_x * stride_w + clamped_y * stride_h + id.x() * stride_c);
                a01 = *reinterpret_cast<T *>(in.ptr() + clamped_x1 * stride_w + clamped_y * stride_h + id.x() * stride_c);
                a10 = *reinterpret_cast<T *>(in.ptr() + clamped_x * stride_w + clamped_y1 * stride_h + id.x() * stride_c);
                a11 = *reinterpret_cast<T *>(in.ptr() + clamped_x1 * stride_w + clamped_y1 * stride_h + id.x() * stride_c);
            }
            else
            {
                a00 = is_valid(offset, 0, input_width - 1, in_yi, 0, input_height - 1) ? *in_ptr : 0;
                a01 = is_valid(offset + 1, 0, input_width - 1, in_yi, 0, input_height - 1) ? *(in_ptr + stride_w_elems) : 0;
                a10 = is_valid(offset, 0, input_width - 1, in_yi + 1, 0, input_height - 1) ? *(in_ptr + stride_h_elems) : 0;
                a11 = is_valid(offset + 1, 0, input_width - 1, in_yi + 1, 0, input_height - 1) ? *(in_ptr + stride_h_elems + stride_w_elems) : 0;
            }

            // Perform interpolation
            const float dx1 = 1.0f - dx_scale;
            const float dy1 = 1.0f - dy_scale;

            const float w1 = dx1 * dy1;
            const float w2 = dx_scale * dy1;
            const float w3 = dx1 * dy_scale;
            const float w4 = dx_scale * dy_scale;

            T res = 0;
            //dequantize quantized input
            if(is_quantized)
            {
                float inp00 = iq_info.dequantize(a00);
                float inp01 = iq_info.dequantize(a01);
                float inp10 = iq_info.dequantize(a10);
                float inp11 = iq_info.dequantize(a11);
                res         = static_cast<T>(oq_info.quantize((inp00 * w1 + inp01 * w2 + inp10 * w3 + inp11 * w4), RoundingPolicy::TO_NEAREST_UP));
            }
            else
            {
                res = static_cast<T>(a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4);
            }
            // Store result
            *reinterpret_cast<T *>(out.ptr()) = res;
        }
        else
        {
            if(border_mode == BorderMode::CONSTANT)
            {
                *reinterpret_cast<T *>(out.ptr()) = border_value;
            }
            else if(border_mode == BorderMode::REPLICATE)
            {
                auto clamped_x                    = utility::clamp<int>(offset, 0, input_width - 1);
                auto clamped_y                    = utility::clamp<int>(in_yi, 0, input_height - 1);
                *reinterpret_cast<T *>(out.ptr()) = *reinterpret_cast<T *>(in.ptr() + clamped_x * stride_w + clamped_y * stride_h + id.x() * stride_c);
            }
        }
    },
    in, out);
}
} // namespace

NEScaleKernel::NEScaleKernel()
    : _func(nullptr), _offsets(nullptr), _dx(nullptr), _dy(nullptr), _input(nullptr), _output(nullptr), _policy(), _border_size(1), _border_mode(), _constant_border_value(PixelValue()),
      _sampling_offset(0), _use_padding(true)
{
}

BorderSize NEScaleKernel::border_size() const
{
    return _border_size;
}

void NEScaleKernel::configure(const ITensor *input, const ITensor *dx, const ITensor *dy, const ITensor *offsets,
                              ITensor *output, InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy,
                              bool use_padding)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  dx != nullptr ? dx->info() : nullptr,
                                                  dy != nullptr ? dy->info() : nullptr,
                                                  offsets != nullptr ? offsets->info() : nullptr,
                                                  output->info(),
                                                  policy, border_mode, constant_border_value, sampling_policy, use_padding));

    // Get data layout and width/height indices
    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    _input                 = input;
    _output                = output;
    _offsets               = offsets;
    _dx                    = dx;
    _dy                    = dy;
    _policy                = policy;
    _border_size           = BorderSize(1);
    _border_mode           = border_mode;
    _constant_border_value = constant_border_value;
    _use_padding           = use_padding;

    if(sampling_policy == SamplingPolicy::CENTER)
    {
        _sampling_offset = 0.5f;
    }

    // Compute the ratio between source width/height and destination width/height
    const auto wr = static_cast<float>(input->info()->dimension(idx_width)) / static_cast<float>(output->info()->dimension(idx_width));
    const auto hr = static_cast<float>(input->info()->dimension(idx_height)) / static_cast<float>(output->info()->dimension(idx_height));

    // Add constant border only on top in case of NHWC layout
    if(data_layout == DataLayout::NHWC)
    {
        _border_size = (border_mode == BorderMode::CONSTANT && policy == InterpolationPolicy::BILINEAR && use_padding) ? BorderSize(1, 0, 0, 0) : BorderSize(0);
    }

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    if(policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
    {
        policy = InterpolationPolicy::NEAREST_NEIGHBOR;
    }

    // Select interpolation function
    switch(policy)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
        {
            _func = (data_layout == DataLayout::NCHW) ? &NEScaleKernel::scale_nearest_nchw : &NEScaleKernel::scale_nhwc;
            break;
        }
        case InterpolationPolicy::BILINEAR:
        {
            _func = (data_layout == DataLayout::NCHW) ? &NEScaleKernel::scale_bilinear_nchw : &NEScaleKernel::scale_nhwc;
            break;
        }
        case InterpolationPolicy::AREA:
        {
            _func = &NEScaleKernel::scale_area_nchw;
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported interpolation mode");
    }

    // Configure window
    std::pair<Status, Window> win_config = validate_and_configure_window(input->info(),
                                                                         dx != nullptr ? dx->info() : nullptr,
                                                                         dy != nullptr ? dy->info() : nullptr,
                                                                         offsets != nullptr ? offsets->info() : nullptr,
                                                                         output->info(),
                                                                         policy, border_mode == BorderMode::UNDEFINED, sampling_policy, border_size(), use_padding);

    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

void NEScaleKernel::scale_nearest_nchw(const Window &window)
{
    const size_t input_stride = _input->info()->strides_in_bytes()[1];

    // Compute the ratio between source height and destination height
    const auto hr = static_cast<float>(_input->info()->dimension(1)) / static_cast<float>(_output->info()->dimension(1));

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

    switch(_input->info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::U8:
        {
            uint8x16_t tmp = vdupq_n_u8(0);

            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto           offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());
                const uint8_t *const in_ptr      = in.ptr();

                const int in_yi         = std::floor((id.y() + 0.5f) * hr);
                const int in_yi_clamped = std::min(static_cast<int>(_input->info()->dimension(1)), std::max(in_yi, -1));
                ARM_COMPUTE_ERROR_ON(in_yi_clamped < -1 || in_yi_clamped > static_cast<int>(_input->info()->dimension(1)));
                const int offset_row = in_yi_clamped * input_stride;

                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[0] + offset_row], tmp, 0);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[1] + offset_row], tmp, 1);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[2] + offset_row], tmp, 2);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[3] + offset_row], tmp, 3);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[4] + offset_row], tmp, 4);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[5] + offset_row], tmp, 5);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[6] + offset_row], tmp, 6);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[7] + offset_row], tmp, 7);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[8] + offset_row], tmp, 8);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[9] + offset_row], tmp, 9);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[10] + offset_row], tmp, 10);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[11] + offset_row], tmp, 11);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[12] + offset_row], tmp, 12);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[13] + offset_row], tmp, 13);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[14] + offset_row], tmp, 14);
                tmp = vsetq_lane_u8(in_ptr[offsets_ptr[15] + offset_row], tmp, 15);

                vst1q_u8(out.ptr(), tmp);
            },
            in, offsets, out);
            break;
        }
        case DataType::S16:
        {
            int16x8x2_t tmp =
            {
                {
                    vdupq_n_s16(0),
                    vdupq_n_s16(0)
                }
            };

            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());

                const int in_yi      = (id.y() + 0.5f) * hr;
                const int offset_row = in_yi * input_stride;

                tmp.val[0] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[0] + offset_row), tmp.val[0], 0);
                tmp.val[0] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[2] + offset_row), tmp.val[0], 1);
                tmp.val[0] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[4] + offset_row), tmp.val[0], 2);
                tmp.val[0] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[6] + offset_row), tmp.val[0], 3);
                tmp.val[0] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[8] + offset_row), tmp.val[0], 4);
                tmp.val[0] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[10] + offset_row), tmp.val[0], 5);
                tmp.val[0] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[12] + offset_row), tmp.val[0], 6);
                tmp.val[0] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[14] + offset_row), tmp.val[0], 7);

                tmp.val[1] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[1] + offset_row), tmp.val[1], 0);
                tmp.val[1] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[3] + offset_row), tmp.val[1], 1);
                tmp.val[1] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[5] + offset_row), tmp.val[1], 2);
                tmp.val[1] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[7] + offset_row), tmp.val[1], 3);
                tmp.val[1] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[9] + offset_row), tmp.val[1], 4);
                tmp.val[1] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[11] + offset_row), tmp.val[1], 5);
                tmp.val[1] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[13] + offset_row), tmp.val[1], 6);
                tmp.val[1] = vsetq_lane_s16(*reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[15] + offset_row), tmp.val[1], 7);

                vst2q_s16(reinterpret_cast<int16_t *>(out.ptr()), tmp);
            },
            in, offsets, out);
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            float16x8x2_t tmp =
            {
                {
                    vdupq_n_f16(0),
                    vdupq_n_f16(0)
                }
            };

            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());

                const int in_yi      = (id.y() + 0.5f) * hr;
                const int offset_row = in_yi * input_stride;

                tmp.val[0] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[0] + offset_row), tmp.val[0], 0);
                tmp.val[0] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[2] + offset_row), tmp.val[0], 1);
                tmp.val[0] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[4] + offset_row), tmp.val[0], 2);
                tmp.val[0] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[6] + offset_row), tmp.val[0], 3);
                tmp.val[0] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[8] + offset_row), tmp.val[0], 4);
                tmp.val[0] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[10] + offset_row), tmp.val[0], 5);
                tmp.val[0] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[12] + offset_row), tmp.val[0], 6);
                tmp.val[0] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[14] + offset_row), tmp.val[0], 7);

                tmp.val[1] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[1] + offset_row), tmp.val[1], 0);
                tmp.val[1] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[3] + offset_row), tmp.val[1], 1);
                tmp.val[1] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[5] + offset_row), tmp.val[1], 2);
                tmp.val[1] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[7] + offset_row), tmp.val[1], 3);
                tmp.val[1] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[9] + offset_row), tmp.val[1], 4);
                tmp.val[1] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[11] + offset_row), tmp.val[1], 5);
                tmp.val[1] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[13] + offset_row), tmp.val[1], 6);
                tmp.val[1] = vsetq_lane_f16(*reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[15] + offset_row), tmp.val[1], 7);

                vst2q_f16(reinterpret_cast<__fp16 *>(out.ptr()), tmp);
            },
            in, offsets, out);
            break;
        }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
        {
            float32x4x4_t tmp =
            {
                {
                    vdupq_n_f32(0),
                    vdupq_n_f32(0),
                    vdupq_n_f32(0),
                    vdupq_n_f32(0)
                }
            };

            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());

                const int in_yi      = (id.y() + 0.5f) * hr;
                const int offset_row = in_yi * input_stride;

                tmp.val[0] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[0] + offset_row), tmp.val[0], 0);
                tmp.val[0] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[4] + offset_row), tmp.val[0], 1);
                tmp.val[0] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[8] + offset_row), tmp.val[0], 2);
                tmp.val[0] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[12] + offset_row), tmp.val[0], 3);

                tmp.val[1] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[1] + offset_row), tmp.val[1], 0);
                tmp.val[1] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[5] + offset_row), tmp.val[1], 1);
                tmp.val[1] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[9] + offset_row), tmp.val[1], 2);
                tmp.val[1] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[13] + offset_row), tmp.val[1], 3);

                tmp.val[2] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[2] + offset_row), tmp.val[2], 0);
                tmp.val[2] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[6] + offset_row), tmp.val[2], 1);
                tmp.val[2] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[10] + offset_row), tmp.val[2], 2);
                tmp.val[2] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[14] + offset_row), tmp.val[2], 3);

                tmp.val[3] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[3] + offset_row), tmp.val[3], 0);
                tmp.val[3] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[7] + offset_row), tmp.val[3], 1);
                tmp.val[3] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[11] + offset_row), tmp.val[3], 2);
                tmp.val[3] = vsetq_lane_f32(*reinterpret_cast<const float *>(in.ptr() + offsets_ptr[15] + offset_row), tmp.val[3], 3);

                vst4q_f32(reinterpret_cast<float *>(out.ptr()), tmp);
            },
            in, offsets, out);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }
}

void NEScaleKernel::scale_bilinear_nchw(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(_input, 1, DataType::U8, DataType::QASYMM8, DataType::S16, DataType::F16, DataType::F32);

    // Compute the ratio between source height and destination height
    const auto hr = static_cast<float>(_input->info()->dimension(1)) / static_cast<float>(_output->info()->dimension(1));

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Window win_off;
    win_off.set(Window::DimX, window.x());
    win_off.set(Window::DimY, window.y());

    for(size_t d = Window::DimZ; d < _offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    Iterator in(_input, win_in);
    Iterator out(_output, window);
    Iterator offsets(_offsets, win_off);
    Iterator dx(_dx, win_off);
    Iterator dy(_dy, win_off);

    /* Input image stride */
    const size_t in_stide_in_bytes = _input->info()->strides_in_bytes()[1];
    const size_t in_stride         = in_stide_in_bytes / _input->info()->element_size();

    const bool             is_quantized = (_input->info()->data_type() == DataType::QASYMM8);
    const QuantizationInfo iq_info      = _input->info()->quantization_info();
    const QuantizationInfo oq_info      = _output->info()->quantization_info();

    switch(_input->info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::U8:
        {
            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());
                const auto dx_ptr      = reinterpret_cast<const float *>(dx.ptr());
                const auto dy_ptr      = reinterpret_cast<const float *>(dy.ptr());
                const auto in_ptr      = reinterpret_cast<const uint8_t *>(in.ptr());

                const int in_yi      = std::floor((id.y() + _sampling_offset) * hr - _sampling_offset);
                const int offset_row = in_yi * in_stide_in_bytes;

                uint8x8_t tmp0 = vdup_n_u8(0);
                if(is_quantized)
                {
                    tmp0 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[0] + offset_row], in_stride, dx_ptr[0], dy_ptr[0], iq_info, oq_info), tmp0, 0);
                    tmp0 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[1] + offset_row], in_stride, dx_ptr[1], dy_ptr[1], iq_info, oq_info), tmp0, 1);
                    tmp0 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[2] + offset_row], in_stride, dx_ptr[2], dy_ptr[2], iq_info, oq_info), tmp0, 2);
                    tmp0 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[3] + offset_row], in_stride, dx_ptr[3], dy_ptr[3], iq_info, oq_info), tmp0, 3);
                    tmp0 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[4] + offset_row], in_stride, dx_ptr[4], dy_ptr[4], iq_info, oq_info), tmp0, 4);
                    tmp0 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[5] + offset_row], in_stride, dx_ptr[5], dy_ptr[5], iq_info, oq_info), tmp0, 5);
                    tmp0 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[6] + offset_row], in_stride, dx_ptr[6], dy_ptr[6], iq_info, oq_info), tmp0, 6);
                    tmp0 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[7] + offset_row], in_stride, dx_ptr[7], dy_ptr[7], iq_info, oq_info), tmp0, 7);
                }
                else
                {
                    tmp0 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[0] + offset_row], in_stride, dx_ptr[0], dy_ptr[0]), tmp0, 0);
                    tmp0 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[1] + offset_row], in_stride, dx_ptr[1], dy_ptr[1]), tmp0, 1);
                    tmp0 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[2] + offset_row], in_stride, dx_ptr[2], dy_ptr[2]), tmp0, 2);
                    tmp0 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[3] + offset_row], in_stride, dx_ptr[3], dy_ptr[3]), tmp0, 3);
                    tmp0 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[4] + offset_row], in_stride, dx_ptr[4], dy_ptr[4]), tmp0, 4);
                    tmp0 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[5] + offset_row], in_stride, dx_ptr[5], dy_ptr[5]), tmp0, 5);
                    tmp0 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[6] + offset_row], in_stride, dx_ptr[6], dy_ptr[6]), tmp0, 6);
                    tmp0 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[7] + offset_row], in_stride, dx_ptr[7], dy_ptr[7]), tmp0, 7);
                }
                uint8x8_t tmp1 = vdup_n_u8(0);
                if(is_quantized)
                {
                    tmp1 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[8] + offset_row], in_stride, dx_ptr[8], dy_ptr[8], iq_info, oq_info), tmp1, 0);
                    tmp1 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[9] + offset_row], in_stride, dx_ptr[9], dy_ptr[9], iq_info, oq_info), tmp1, 1);
                    tmp1 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[10] + offset_row], in_stride, dx_ptr[10], dy_ptr[10], iq_info, oq_info), tmp1, 2);
                    tmp1 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[11] + offset_row], in_stride, dx_ptr[11], dy_ptr[11], iq_info, oq_info), tmp1, 3);
                    tmp1 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[12] + offset_row], in_stride, dx_ptr[12], dy_ptr[12], iq_info, oq_info), tmp1, 4);
                    tmp1 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[13] + offset_row], in_stride, dx_ptr[13], dy_ptr[13], iq_info, oq_info), tmp1, 5);
                    tmp1 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[14] + offset_row], in_stride, dx_ptr[14], dy_ptr[14], iq_info, oq_info), tmp1, 6);
                    tmp1 = vset_lane_u8(delta_bilinear_c1_quantized(&in_ptr[offsets_ptr[15] + offset_row], in_stride, dx_ptr[15], dy_ptr[15], iq_info, oq_info), tmp1, 7);
                }
                else
                {
                    tmp1 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[8] + offset_row], in_stride, dx_ptr[8], dy_ptr[8]), tmp1, 0);
                    tmp1 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[9] + offset_row], in_stride, dx_ptr[9], dy_ptr[9]), tmp1, 1);
                    tmp1 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[10] + offset_row], in_stride, dx_ptr[10], dy_ptr[10]), tmp1, 2);
                    tmp1 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[11] + offset_row], in_stride, dx_ptr[11], dy_ptr[11]), tmp1, 3);
                    tmp1 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[12] + offset_row], in_stride, dx_ptr[12], dy_ptr[12]), tmp1, 4);
                    tmp1 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[13] + offset_row], in_stride, dx_ptr[13], dy_ptr[13]), tmp1, 5);
                    tmp1 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[14] + offset_row], in_stride, dx_ptr[14], dy_ptr[14]), tmp1, 6);
                    tmp1 = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[15] + offset_row], in_stride, dx_ptr[15], dy_ptr[15]), tmp1, 7);
                }
                vst1q_u8(out.ptr(), vcombine_u8(tmp0, tmp1));
            },
            in, offsets, dx, dy, out);
            break;
        }
        case DataType::S16:
        {
            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());
                const auto dx_ptr      = reinterpret_cast<const float *>(dx.ptr());
                const auto dy_ptr      = reinterpret_cast<const float *>(dy.ptr());

                const int in_yi      = std::floor((id.y() + _sampling_offset) * hr - _sampling_offset);
                const int offset_row = in_yi * in_stide_in_bytes;

                int16x8x2_t tmp =
                {
                    {
                        vdupq_n_s16(0),
                        vdupq_n_s16(0)
                    }
                };

                tmp.val[0] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[0] + offset_row), in_stride, dx_ptr[0], dy_ptr[0]), tmp.val[0], 0);
                tmp.val[0] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[2] + offset_row), in_stride, dx_ptr[2], dy_ptr[2]), tmp.val[0], 1);
                tmp.val[0] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[4] + offset_row), in_stride, dx_ptr[4], dy_ptr[4]), tmp.val[0], 2);
                tmp.val[0] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[6] + offset_row), in_stride, dx_ptr[6], dy_ptr[6]), tmp.val[0], 3);
                tmp.val[0] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[8] + offset_row), in_stride, dx_ptr[8], dy_ptr[8]), tmp.val[0], 4);
                tmp.val[0] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[10] + offset_row), in_stride, dx_ptr[10], dy_ptr[10]), tmp.val[0], 5);
                tmp.val[0] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[12] + offset_row), in_stride, dx_ptr[12], dy_ptr[12]), tmp.val[0], 6);
                tmp.val[0] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[14] + offset_row), in_stride, dx_ptr[14], dy_ptr[14]), tmp.val[0], 7);

                tmp.val[1] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[1] + offset_row), in_stride, dx_ptr[1], dy_ptr[1]), tmp.val[1], 0);
                tmp.val[1] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[3] + offset_row), in_stride, dx_ptr[3], dy_ptr[3]), tmp.val[1], 1);
                tmp.val[1] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[5] + offset_row), in_stride, dx_ptr[5], dy_ptr[5]), tmp.val[1], 2);
                tmp.val[1] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[7] + offset_row), in_stride, dx_ptr[7], dy_ptr[7]), tmp.val[1], 3);
                tmp.val[1] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[9] + offset_row), in_stride, dx_ptr[9], dy_ptr[9]), tmp.val[1], 4);
                tmp.val[1] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[11] + offset_row), in_stride, dx_ptr[11], dy_ptr[11]), tmp.val[1], 5);
                tmp.val[1] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[13] + offset_row), in_stride, dx_ptr[13], dy_ptr[13]), tmp.val[1], 6);
                tmp.val[1] = vsetq_lane_s16(delta_bilinear_c1(reinterpret_cast<const int16_t *>(in.ptr() + offsets_ptr[15] + offset_row), in_stride, dx_ptr[15], dy_ptr[15]), tmp.val[1], 7);

                vst2q_s16(reinterpret_cast<int16_t *>(out.ptr()), tmp);
            },
            in, offsets, dx, dy, out);
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());
                const auto dx_ptr      = reinterpret_cast<const float *>(dx.ptr());
                const auto dy_ptr      = reinterpret_cast<const float *>(dy.ptr());

                const int in_yi      = std::floor((id.y() + _sampling_offset) * hr - _sampling_offset);
                const int offset_row = in_yi * in_stide_in_bytes;

                float16x8x2_t tmp =
                {
                    {
                        vdupq_n_f16(0),
                        vdupq_n_f16(0)
                    }
                };

                tmp.val[0] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[0] + offset_row), in_stride, dx_ptr[0], dy_ptr[0]), tmp.val[0], 0);
                tmp.val[0] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[2] + offset_row), in_stride, dx_ptr[2], dy_ptr[2]), tmp.val[0], 1);
                tmp.val[0] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[4] + offset_row), in_stride, dx_ptr[4], dy_ptr[4]), tmp.val[0], 2);
                tmp.val[0] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[6] + offset_row), in_stride, dx_ptr[6], dy_ptr[6]), tmp.val[0], 3);
                tmp.val[0] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[8] + offset_row), in_stride, dx_ptr[8], dy_ptr[8]), tmp.val[0], 4);
                tmp.val[0] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[10] + offset_row), in_stride, dx_ptr[10], dy_ptr[10]), tmp.val[0], 5);
                tmp.val[0] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[12] + offset_row), in_stride, dx_ptr[12], dy_ptr[12]), tmp.val[0], 6);
                tmp.val[0] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[14] + offset_row), in_stride, dx_ptr[14], dy_ptr[14]), tmp.val[0], 7);

                tmp.val[1] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[1] + offset_row), in_stride, dx_ptr[1], dy_ptr[1]), tmp.val[1], 0);
                tmp.val[1] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[3] + offset_row), in_stride, dx_ptr[3], dy_ptr[3]), tmp.val[1], 1);
                tmp.val[1] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[5] + offset_row), in_stride, dx_ptr[5], dy_ptr[5]), tmp.val[1], 2);
                tmp.val[1] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[7] + offset_row), in_stride, dx_ptr[7], dy_ptr[7]), tmp.val[1], 3);
                tmp.val[1] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[9] + offset_row), in_stride, dx_ptr[9], dy_ptr[9]), tmp.val[1], 4);
                tmp.val[1] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[11] + offset_row), in_stride, dx_ptr[11], dy_ptr[11]), tmp.val[1], 5);
                tmp.val[1] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[13] + offset_row), in_stride, dx_ptr[13], dy_ptr[13]), tmp.val[1], 6);
                tmp.val[1] = vsetq_lane_f16(delta_bilinear_c1(reinterpret_cast<const __fp16 *>(in.ptr() + offsets_ptr[15] + offset_row), in_stride, dx_ptr[15], dy_ptr[15]), tmp.val[1], 7);

                vst2q_f16(reinterpret_cast<__fp16 *>(out.ptr()), tmp);
            },
            in, offsets, dx, dy, out);
            break;
        }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
        {
            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());
                const auto dx_ptr      = reinterpret_cast<const float *>(dx.ptr());
                const auto dy_ptr      = reinterpret_cast<const float *>(dy.ptr());

                const int in_yi      = std::floor((id.y() + _sampling_offset) * hr - _sampling_offset);
                const int offset_row = in_yi * in_stide_in_bytes;

                float32x4x4_t tmp =
                {
                    {
                        vdupq_n_f32(0),
                        vdupq_n_f32(0),
                        vdupq_n_f32(0),
                        vdupq_n_f32(0)
                    }
                };

                tmp.val[0] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[0] + offset_row), in_stride, dx_ptr[0], dy_ptr[0]), tmp.val[0], 0);
                tmp.val[0] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[4] + offset_row), in_stride, dx_ptr[4], dy_ptr[4]), tmp.val[0], 1);
                tmp.val[0] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[8] + offset_row), in_stride, dx_ptr[8], dy_ptr[8]), tmp.val[0], 2);
                tmp.val[0] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[12] + offset_row), in_stride, dx_ptr[12], dy_ptr[12]), tmp.val[0], 3);

                tmp.val[1] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[1] + offset_row), in_stride, dx_ptr[1], dy_ptr[1]), tmp.val[1], 0);
                tmp.val[1] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[5] + offset_row), in_stride, dx_ptr[5], dy_ptr[5]), tmp.val[1], 1);
                tmp.val[1] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[9] + offset_row), in_stride, dx_ptr[9], dy_ptr[9]), tmp.val[1], 2);
                tmp.val[1] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[13] + offset_row), in_stride, dx_ptr[13], dy_ptr[13]), tmp.val[1], 3);

                tmp.val[2] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[2] + offset_row), in_stride, dx_ptr[2], dy_ptr[2]), tmp.val[2], 0);
                tmp.val[2] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[6] + offset_row), in_stride, dx_ptr[6], dy_ptr[6]), tmp.val[2], 1);
                tmp.val[2] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[10] + offset_row), in_stride, dx_ptr[10], dy_ptr[10]), tmp.val[2], 2);
                tmp.val[2] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[14] + offset_row), in_stride, dx_ptr[14], dy_ptr[14]), tmp.val[2], 3);

                tmp.val[3] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[3] + offset_row), in_stride, dx_ptr[3], dy_ptr[3]), tmp.val[3], 0);
                tmp.val[3] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[7] + offset_row), in_stride, dx_ptr[7], dy_ptr[7]), tmp.val[3], 1);
                tmp.val[3] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[11] + offset_row), in_stride, dx_ptr[11], dy_ptr[11]), tmp.val[3], 2);
                tmp.val[3] = vsetq_lane_f32(delta_bilinear_c1(reinterpret_cast<const float *>(in.ptr() + offsets_ptr[15] + offset_row), in_stride, dx_ptr[15], dy_ptr[15]), tmp.val[3], 3);

                vst4q_f32(reinterpret_cast<float *>(out.ptr()), tmp);
            },
            in, offsets, dx, dy, out);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }
}

void NEScaleKernel::scale_area_nchw(const Window &window)
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

    const auto   wr        = static_cast<float>(_input->info()->dimension(0)) / static_cast<float>(_output->info()->dimension(0));
    const auto   hr        = static_cast<float>(_input->info()->dimension(1)) / static_cast<float>(_output->info()->dimension(1));
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

void NEScaleKernel::scale_nhwc(const Window &window)
{
    // Get data layout and width/height indices
    const DataLayout data_layout  = _input->info()->data_layout();
    const int        idx_channels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const int        idx_width    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    const size_t input_stride_w = _input->info()->strides_in_bytes()[idx_width];
    const size_t input_stride_h = _input->info()->strides_in_bytes()[idx_height];
    const size_t input_stride_c = _input->info()->strides_in_bytes()[idx_channels];

    // Compute the ratio between source height and destination height
    const auto hr = static_cast<float>(_input->info()->dimension(idx_height)) / static_cast<float>(_output->info()->dimension(idx_height));

    // Don't increment in width/height/channels for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    switch(_input->info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::U8:
        {
            if(_policy == InterpolationPolicy::NEAREST_NEIGHBOR)
            {
                scale_nearest_nhwc_core<uint8_t>(_input, _offsets, _output, hr, window, win_in, input_stride_w, input_stride_h, input_stride_c);
            }
            else
            {
                scale_bilinear_nhwc_core<uint8_t, uint8_t>(_input, _offsets, _dx, _dy, _output, hr, _sampling_offset,
                                                           window, win_in, input_stride_w, input_stride_h, input_stride_c, _border_mode, _constant_border_value, _use_padding);
            }
            break;
        }
        case DataType::S16:
        {
            if(_policy == InterpolationPolicy::NEAREST_NEIGHBOR)
            {
                scale_nearest_nhwc_core<int16_t>(_input, _offsets, _output, hr, window, win_in, input_stride_w, input_stride_h, input_stride_c);
            }
            else
            {
                scale_bilinear_nhwc_core<int16_t, int16_t>(_input, _offsets, _dx, _dy, _output, hr, _sampling_offset,
                                                           window, win_in, input_stride_w, input_stride_h, input_stride_c, _border_mode, _constant_border_value, _use_padding);
            }
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            if(_policy == InterpolationPolicy::NEAREST_NEIGHBOR)
            {
                scale_nearest_nhwc_core<float16_t>(_input, _offsets, _output, hr,
                                                   window, win_in, input_stride_w, input_stride_h, input_stride_c);
            }
            else
            {
                scale_bilinear_nhwc_core<float16_t, half>(_input, _offsets, _dx, _dy, _output, hr, _sampling_offset,
                                                          window, win_in, input_stride_w, input_stride_h, input_stride_c, _border_mode, _constant_border_value, _use_padding);
            }
            break;
        }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
        {
            if(_policy == InterpolationPolicy::NEAREST_NEIGHBOR)
            {
                scale_nearest_nhwc_core<float>(_input, _offsets, _output, hr, window, win_in, input_stride_w, input_stride_h, input_stride_c);
            }
            else
            {
                scale_bilinear_nhwc_core<float, float>(_input, _offsets, _dx, _dy, _output, hr, _sampling_offset,
                                                       window, win_in, input_stride_w, input_stride_h, input_stride_c, _border_mode, _constant_border_value, _use_padding);
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported");
            break;
    }
}

Status NEScaleKernel::validate(const ITensorInfo *input, const ITensorInfo *dx, const ITensorInfo *dy,
                               const ITensorInfo *offsets, ITensorInfo *output, InterpolationPolicy policy,
                               BorderMode border_mode, PixelValue constant_border_value, SamplingPolicy sampling_policy, bool use_padding)
{
    BorderSize border_size(1);
    if(input->data_layout() == DataLayout::NHWC)
    {
        border_size = (border_mode == BorderMode::CONSTANT && policy == InterpolationPolicy::BILINEAR) ? BorderSize(1, 0, 0, 0) : BorderSize(0);
    }

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, dx, dy, offsets, output, policy, border_mode, constant_border_value, sampling_policy, use_padding));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(),
                                                              dx != nullptr ? dx->clone().get() : nullptr,
                                                              dy != nullptr ? dy->clone().get() : nullptr,
                                                              offsets != nullptr ? offsets->clone().get() : nullptr,
                                                              output->clone().get(),
                                                              policy, border_mode == BorderMode::UNDEFINED, sampling_policy, border_size, use_padding)
                                .first);

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
