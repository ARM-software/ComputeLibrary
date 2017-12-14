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
#include "arm_compute/core/NEON/kernels/NEScaleKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

NEScaleKernel::NEScaleKernel()
    : _func(nullptr), _offsets(nullptr), _dx(nullptr), _dy(nullptr), _input(nullptr), _output(nullptr)
{
}

BorderSize NEScaleKernel::border_size() const
{
    return BorderSize(1);
}

void NEScaleKernel::configure(const ITensor *input, const ITensor *dx, const ITensor *dy, const ITensor *offsets, ITensor *output, InterpolationPolicy policy, bool border_undefined,
                              SamplingPolicy sampling_policy)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON(output == input);
    ARM_COMPUTE_ERROR_ON(sampling_policy != SamplingPolicy::CENTER);
    ARM_COMPUTE_UNUSED(sampling_policy);

    if(policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(offsets, 1, DataType::S32);
    }

    if(policy == InterpolationPolicy::BILINEAR)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(offsets, 1, DataType::S32);
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dx, 1, DataType::F32);
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dy, 1, DataType::F32);
    }

    ARM_COMPUTE_ERROR_ON(output->info()->dimension(0) == 0);
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(1) == 0);

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input   = input;
    _output  = output;
    _offsets = offsets;
    _dx      = dx;
    _dy      = dy;

    /* Compute the ratio between source width/height and destination width/height */
    const auto wr = static_cast<float>(input->info()->dimension(0)) / static_cast<float>(output->info()->dimension(0));
    const auto hr = static_cast<float>(input->info()->dimension(1)) / static_cast<float>(output->info()->dimension(1));

    /* Area interpolation behaves as Nearest Neighbour in case of up-sampling */
    if(policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
    {
        policy = InterpolationPolicy::NEAREST_NEIGHBOR;
    }

    switch(policy)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
        {
            _func = &NEScaleKernel::scale_nearest;
            break;
        }
        case InterpolationPolicy::BILINEAR:
        {
            ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(_dx, 1, DataType::F32);
            ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(_dy, 1, DataType::F32);

            _func = &NEScaleKernel::scale_bilinear;
            break;
        }
        case InterpolationPolicy::AREA:
        {
            _func = &NEScaleKernel::scale_area;
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported interpolation mode");
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    const ValidRegion &input_valid_region = input->info()->valid_region();

    // Reads can occur within the valid region of the input
    AccessWindowStatic input_access(input->info(),
                                    input_valid_region.anchor[0] - border_size().left, input_valid_region.anchor[1] - border_size().top,
                                    input_valid_region.anchor[0] + input_valid_region.shape[0] + border_size().right,
                                    input_valid_region.anchor[1] + input_valid_region.shape[1] + border_size().bottom);
    AccessWindowHorizontal offsets_access(offsets == nullptr ? nullptr : offsets->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal dx_access(dx == nullptr ? nullptr : dx->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal dy_access(dy == nullptr ? nullptr : dy->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              input_access,
                              offsets_access,
                              dx_access,
                              dy_access,
                              output_access);

    output_access.set_valid_region(win, calculate_valid_region_scale(*(input->info()), output->info()->tensor_shape(), policy, border_size(), border_undefined));
    INEKernel::configure(win);
}

void NEScaleKernel::scale_nearest(const Window &window)
{
    const size_t input_stride = _input->info()->strides_in_bytes()[1];

    // Compute the ratio between source height and destination height
    const auto hr = static_cast<float>(_input->info()->dimension(1)) / static_cast<float>(_output->info()->dimension(1));

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Window win_off;
    win_off.set(Window::DimX, window[Window::DimX]);
    win_off.set(Window::DimY, window[Window::DimY]);

    for(size_t d = Window::DimZ; d < _offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    Iterator in(_input, win_in);
    Iterator out(_output, window);
    Iterator offsets(_offsets, win_off);

    switch(_input->info()->data_type())
    {
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

void NEScaleKernel::scale_bilinear(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(_input, 1, DataType::U8, DataType::S16, DataType::F32);

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

    switch(_input->info()->data_type())
    {
        case DataType::U8:
        {
            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());
                const auto dx_ptr      = reinterpret_cast<const float *>(dx.ptr());
                const auto dy_ptr      = reinterpret_cast<const float *>(dy.ptr());
                const auto in_ptr      = reinterpret_cast<const uint8_t *>(in.ptr());

                const int in_yi      = std::floor((id.y() + 0.5f) * hr - 0.5f);
                const int offset_row = in_yi * in_stide_in_bytes;

                uint8x8_t tmp0 = vdup_n_u8(0);
                tmp0           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[0] + offset_row], in_stride, dx_ptr[0], dy_ptr[0]), tmp0, 0);
                tmp0           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[1] + offset_row], in_stride, dx_ptr[1], dy_ptr[1]), tmp0, 1);
                tmp0           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[2] + offset_row], in_stride, dx_ptr[2], dy_ptr[2]), tmp0, 2);
                tmp0           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[3] + offset_row], in_stride, dx_ptr[3], dy_ptr[3]), tmp0, 3);
                tmp0           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[4] + offset_row], in_stride, dx_ptr[4], dy_ptr[4]), tmp0, 4);
                tmp0           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[5] + offset_row], in_stride, dx_ptr[5], dy_ptr[5]), tmp0, 5);
                tmp0           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[6] + offset_row], in_stride, dx_ptr[6], dy_ptr[6]), tmp0, 6);
                tmp0           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[7] + offset_row], in_stride, dx_ptr[7], dy_ptr[7]), tmp0, 7);

                uint8x8_t tmp1 = vdup_n_u8(0);
                tmp1           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[8] + offset_row], in_stride, dx_ptr[8], dy_ptr[8]), tmp1, 0);
                tmp1           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[9] + offset_row], in_stride, dx_ptr[9], dy_ptr[9]), tmp1, 1);
                tmp1           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[10] + offset_row], in_stride, dx_ptr[10], dy_ptr[10]), tmp1, 2);
                tmp1           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[11] + offset_row], in_stride, dx_ptr[11], dy_ptr[11]), tmp1, 3);
                tmp1           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[12] + offset_row], in_stride, dx_ptr[12], dy_ptr[12]), tmp1, 4);
                tmp1           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[13] + offset_row], in_stride, dx_ptr[13], dy_ptr[13]), tmp1, 5);
                tmp1           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[14] + offset_row], in_stride, dx_ptr[14], dy_ptr[14]), tmp1, 6);
                tmp1           = vset_lane_u8(delta_bilinear_c1(&in_ptr[offsets_ptr[15] + offset_row], in_stride, dx_ptr[15], dy_ptr[15]), tmp1, 7);

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

                const int in_yi      = std::floor((id.y() + 0.5f) * hr - 0.5f);
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
        case DataType::F32:
        {
            execute_window_loop(window, [&](const Coordinates & id)
            {
                const auto offsets_ptr = reinterpret_cast<const int32_t *>(offsets.ptr());
                const auto dx_ptr      = reinterpret_cast<const float *>(dx.ptr());
                const auto dy_ptr      = reinterpret_cast<const float *>(dy.ptr());

                const int in_yi      = std::floor((id.y() + 0.5f) * hr - 0.5f);
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

void NEScaleKernel::scale_area(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(_input, 1, DataType::U8);

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

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

void NEScaleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
