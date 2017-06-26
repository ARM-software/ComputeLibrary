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
#include "arm_compute/core/NEON/kernels/NEPoolingLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <arm_neon.h>
#include <limits>
#include <set>
#include <string>
#include <tuple>

using namespace arm_compute;

namespace
{
inline float calculate_avg_scale(const Coordinates &id, const int pool_size, const int upper_bound_w, const int upper_bound_h,
                                 const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    const int start_x = id.x() * stride_x - pad_x;
    const int start_y = id.y() * stride_y - pad_y;
    const int end_x   = std::min(start_x + pool_size, upper_bound_w);
    const int end_y   = std::min(start_y + pool_size, upper_bound_h);
    return 1.f / ((end_y - start_y) * (end_x - start_x));
}

inline qint8_t calculate_avg_scale_q8(const Coordinates &id, int pool_size, int upper_bound_w, int upper_bound_h,
                                      int pad_x, int pad_y, int stride_x, int stride_y, int fixed_point_position)
{
    static const std::array<qint8_t, 10> scale_values_q8 =
    { { 0x0, 0x0, 0x40, 0x2A, 0x20, 0x19, 0x15, 0x12, 0x10, 0xE } };
    const int start_x = id.x() * stride_x - pad_x;
    const int start_y = id.y() * stride_y - pad_y;
    const int end_x   = std::min(start_x + pool_size, upper_bound_w);
    const int end_y   = std::min(start_y + pool_size, upper_bound_h);
    const int val     = ((end_y - start_y) * (end_x - start_x));
    return scale_values_q8[val] >> (7 - fixed_point_position);
}
} // namespace

NEPoolingLayerKernel::NEPoolingLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _pool_info(), _num_elems_processed_per_iteration(0), _border_size(0)
{
}

BorderSize NEPoolingLayerKernel::border_size() const
{
    return _border_size;
}

void NEPoolingLayerKernel::configure(const ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info)
{
    int                 pool_pad_x      = 0;
    int                 pool_pad_y      = 0;
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    unsigned int        pooled_w        = 0;
    unsigned int        pooled_h        = 0;
    PoolingType         pool_type       = pool_info.pool_type();
    int                 pool_size       = pool_info.pool_size();
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info();
    std::tie(pool_pad_x, pool_pad_y)       = pad_stride_info.pad();
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();

    static const std::set<int> supported_pool_sizes = { 2, 3, 7 };
    ARM_COMPUTE_UNUSED(supported_pool_sizes);

    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QS8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON(supported_pool_sizes.find(pool_size) == supported_pool_sizes.end());
    ARM_COMPUTE_ERROR_ON(7 == pool_size && input->info()->data_type() != DataType::F32);
    ARM_COMPUTE_ERROR_ON(pool_pad_x >= pool_size || pool_pad_y >= pool_size);
    ARM_COMPUTE_ERROR_ON(input->info()->data_type() == DataType::QS8 && pool_type == PoolingType::AVG && input->info()->fixed_point_position() > 6);
    ARM_COMPUTE_ERROR_ON(input->info()->data_type() == DataType::QS8 && pool_stride_x > 2);

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1),
                                                     pool_size, pool_size, pool_info.pad_stride_info());

    // Output auto initialization if not yet initialized
    {
        TensorShape output_shape{ input->info()->tensor_shape() };
        output_shape.set(0, pooled_w);
        output_shape.set(1, pooled_h);

        auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());
    }

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) != pooled_w) || (output->info()->dimension(1) != pooled_h));

    unsigned int num_elems_read_per_iteration      = 0;
    unsigned int num_elems_processed_per_iteration = 0;
    unsigned int num_elems_horizontal_window       = 0;

    // Select element size
    switch(input->info()->data_type())
    {
        case DataType::QS8:
            num_elems_read_per_iteration = 16;
            switch(pool_size)
            {
                case 2:
                    num_elems_processed_per_iteration = 8;
                    break;
                case 3:
                    num_elems_processed_per_iteration = 7;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Pooling size not supported");
                    break;
            }
            num_elems_horizontal_window = 8;
            break;
#ifdef ARM_COMPUTE_ENABLE_FP16
        case DataType::F16:
            switch(pool_size)
            {
                case 2:
                    num_elems_read_per_iteration      = 16;
                    num_elems_processed_per_iteration = 8;
                    num_elems_horizontal_window       = 8;
                    break;
                case 3:
                    num_elems_read_per_iteration      = 4;
                    num_elems_processed_per_iteration = 1;
                    num_elems_horizontal_window       = 1;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Pooling size not supported");
                    break;
            }
            break;
#endif /* ARM_COMPUTE_ENABLE_FP16 */
        case DataType::F32:
            switch(pool_size)
            {
                case 2:
                    num_elems_read_per_iteration = 2;
                    break;
                case 3:
                    num_elems_read_per_iteration = 4; // We use vload4 for pooling3
                    break;
                case 7:
                    num_elems_read_per_iteration = 8; // We use vload8 for pooling7
                    break;
                default:
                    ARM_COMPUTE_ERROR("Pooling size not supported");
                    break;
            }
            num_elems_processed_per_iteration = 1;
            num_elems_horizontal_window       = 1;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    _num_elems_processed_per_iteration = num_elems_processed_per_iteration;
    const int input_width              = input->info()->dimension(0);
    const int input_height             = input->info()->dimension(1);
    const int upper_bound_w            = ((pooled_w - 1) * pool_stride_x - pool_pad_x + num_elems_read_per_iteration) - input_width;
    const int upper_bound_h            = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

    // Set instance variables
    _input              = input;
    _output             = output;
    _pool_info          = pool_info;
    _border_size        = BorderSize(pool_pad_y, pool_pad_x);
    _border_size.right  = std::max(upper_bound_w, pool_pad_x);
    _border_size.bottom = std::max(upper_bound_h, pool_pad_y);

    // Select appropriate function
    switch(pool_size)
    {
        case 2:
            if(input->info()->data_type() == DataType::QS8)
            {
                _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling2_q8<PoolingType::AVG> : &NEPoolingLayerKernel::pooling2_q8<PoolingType::MAX>;
            }
            else if(input->info()->data_type() == DataType::F16)
            {
                _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling2_f16<PoolingType::AVG> : &NEPoolingLayerKernel::pooling2_f16<PoolingType::MAX>;
            }
            else if(input->info()->data_type() == DataType::F32)
            {
                _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling2_f32<PoolingType::AVG> : &NEPoolingLayerKernel::pooling2_f32<PoolingType::MAX>;
            }
            break;
        case 3:
            if(input->info()->data_type() == DataType::QS8)
            {
                _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling3_q8<PoolingType::AVG> : &NEPoolingLayerKernel::pooling3_q8<PoolingType::MAX>;
            }
            else if(input->info()->data_type() == DataType::F16)
            {
                _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling3_f16<PoolingType::AVG> : &NEPoolingLayerKernel::pooling3_f16<PoolingType::MAX>;
            }
            else if(input->info()->data_type() == DataType::F32)
            {
                _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling3_f32<PoolingType::AVG> : &NEPoolingLayerKernel::pooling3_f32<PoolingType::MAX>;
            }
            break;
        case 7:
            _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling7_f32<PoolingType::AVG> : &NEPoolingLayerKernel::pooling7_f32<PoolingType::MAX>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported pooling size");
            break;
    }

    // Configure kernel window
    Window                 win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowStatic     input_access(input->info(), -pool_pad_x, -pool_pad_y, input_width + _border_size.right, input_height + _border_size.bottom);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_horizontal_window);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
    INEKernel::configure(win);
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling2_q8(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    const int     fixed_point_position = _input->info()->fixed_point_position();
    constexpr int pool_size            = 2;
    int           pool_pad_x           = 0;
    int           pool_pad_y           = 0;
    int           pool_stride_x        = 0;
    int           pool_stride_y        = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + pool_pad_x;
    const int upper_bound_h = _input->info()->dimension(1) + pool_pad_y;

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = vld1q_qs8(reinterpret_cast<const qint8_t *>(input_top_ptr + input.offset()));
        const auto bottom_data = vld1q_qs8(reinterpret_cast<const qint8_t *>(input_bottom_ptr + input.offset()));
        qint8x8_t  res         = {};
        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            const qint8_t   scale     = calculate_avg_scale_q8(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y, fixed_point_position);
            const qint8x8_t scale_vec = vdup_n_qs8(scale);

            // Perform pooling
            const qint8x16_t sum_data = vqaddq_qs8(top_data, bottom_data);
            res                       = vqmul_qs8(vpadd_s8(vget_low_s8(sum_data), vget_high_s8(sum_data)), scale_vec, fixed_point_position);
        }
        else
        {
            const qint8x16_t max_data = vmaxq_s8(top_data, bottom_data);
            res                       = vpmax_s8(vget_low_s8(max_data), vget_high_s8(max_data));
        }
        vst1_qs8(reinterpret_cast<qint8_t *>(output.ptr()), res);
    },
    input, output);
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling3_f16(const Window &window_input, const Window &window)
{
#ifdef ARM_COMPUTE_ENABLE_FP16
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr const int pool_size     = 3;
    int                 pool_pad_x    = 0;
    int                 pool_pad_y    = 0;
    int                 pool_stride_x = 0;
    int                 pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + pool_pad_x;
    const int upper_bound_h = _input->info()->dimension(1) + pool_pad_y;

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const unsigned char *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float16x4_t top_data    = vld1_f16(reinterpret_cast<const float16_t *>(input_top_ptr + input.offset()));
        const float16x4_t middle_data = vld1_f16(reinterpret_cast<const float16_t *>(input_middle_ptr + input.offset()));
        const float16x4_t bottom_data = vld1_f16(reinterpret_cast<const float16_t *>(input_bottom_ptr + input.offset()));
        float16x4_t       res         = {};
        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            const float       scale   = calculate_avg_scale(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
            const float16x4_t scale_v = vdup_n_f16(scale);
            // Perform pooling
            const float16x4_t sum_data = vadd_f16(vadd_f16(top_data, bottom_data), middle_data);
            res                        = vpadd_f16(vset_lane_f16(0.f, sum_data, 3), sum_data);
            res                        = vmul_f16(vpadd_f16(res, res), scale_v);
        }
        else
        {
            const float16x4_t max_data = vmax_f16(vmax_f16(top_data, bottom_data), middle_data);
            res                        = vpmax_f16(vset_lane_f16(-std::numeric_limits<float>::max(), max_data, 3), max_data);
            res                        = vpmax_f16(res, res);
        }
        *(reinterpret_cast<float16_t *>(output.ptr())) = vget_lane_f16(res, 0);
    },
    input, output);
#else  /* ARM_COMPUTE_ENABLE_FP16 */
    ARM_COMPUTE_UNUSED(window_input);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR("FP16 Not supported! Recompile the library with arch=arm64-v8.2-a");
#endif /* ARM_COMPUTE_ENABLE_FP16 */
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling2_f16(const Window &window_input, const Window &window)
{
#ifdef ARM_COMPUTE_ENABLE_FP16
    Iterator      input(_input, window_input);
    Iterator      output(_output, window);
    constexpr int pool_size = 2;
    int           pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + pool_pad_x;
    const int upper_bound_h = _input->info()->dimension(1) + pool_pad_y;

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto  top_data    = vld2q_f16(reinterpret_cast<const float16_t *>(input_top_ptr + input.offset()));
        const auto  bottom_data = vld2q_f16(reinterpret_cast<const float16_t *>(input_bottom_ptr + input.offset()));
        float16x8_t res         = {};

        if(pooling_type == PoolingType::AVG)
        {
            const float       scale   = calculate_avg_scale(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
            const float16x8_t scale_v = vdupq_n_f16(scale);
            res                       = vmulq_f16(scale_v, vaddq_f16(bottom_data.val[1], vaddq_f16(bottom_data.val[0], vaddq_f16(top_data.val[0], top_data.val[1]))));
        }
        else
        {
            res = vmaxq_f16(bottom_data.val[1], vmaxq_f16(bottom_data.val[0], vmaxq_f16(top_data.val[0], top_data.val[1])));
        }
        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), res);
    },
    input, output);
#else  /* ARM_COMPUTE_ENABLE_FP16 */
    ARM_COMPUTE_UNUSED(window_input);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR("FP16 Not supported! Recompile the library with arch=arm64-v8.2-a");
#endif /* ARM_COMPUTE_ENABLE_FP16 */
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling2_f32(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr int pool_size     = 2;
    int           pool_pad_x    = 0;
    int           pool_pad_y    = 0;
    int           pool_stride_x = 0;
    int           pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + pool_pad_x;
    const int upper_bound_h = _input->info()->dimension(1) + pool_pad_y;

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float32x2_t top_data    = vld1_f32(reinterpret_cast<const float *>(input_top_ptr + input.offset()));
        const float32x2_t bottom_data = vld1_f32(reinterpret_cast<const float *>(input_bottom_ptr + input.offset()));
        float32x2_t       res         = {};
        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            float             scale   = calculate_avg_scale(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
            const float32x2_t scale_v = vdup_n_f32(scale);

            // Perform pooling
            const float32x2_t sum_data = vadd_f32(top_data, bottom_data);
            res                        = vmul_f32(vpadd_f32(sum_data, sum_data), scale_v);
        }
        else
        {
            const float32x2_t max_data = vmax_f32(top_data, bottom_data);
            res                        = vpmax_f32(max_data, max_data);
        }
        *(reinterpret_cast<float *>(output.ptr())) = vget_lane_f32(res, 0);
    },
    input, output);
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling3_q8(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    const int     fixed_point_position = _input->info()->fixed_point_position();
    constexpr int pool_size            = 3;
    int           pool_pad_x           = 0;
    int           pool_pad_y           = 0;
    int           pool_stride_x        = 0;
    int           pool_stride_y        = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + pool_pad_x;
    const int upper_bound_h = _input->info()->dimension(1) + pool_pad_y;

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const uint8_t *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = vld1q_qs8(reinterpret_cast<const qint8_t *>(input_top_ptr + input.offset()));
        const auto middle_data = vld1q_qs8(reinterpret_cast<const qint8_t *>(input_middle_ptr + input.offset()));
        const auto bottom_data = vld1q_qs8(reinterpret_cast<const qint8_t *>(input_bottom_ptr + input.offset()));
        qint8x8_t  res         = {};
        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            const qint8_t   scale     = calculate_avg_scale_q8(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y, fixed_point_position);
            const qint8x8_t scale_vec = vdup_n_qs8(scale);

            // Perform pooling for stride 2
            const qint8x16_t sum_data  = vqaddq_qs8(vqaddq_qs8(top_data, bottom_data), middle_data);
            const qint8x16_t sum_data2 = vextq_s8(sum_data, sum_data, 1);
            const qint8x16_t sum_data3 = vextq_s8(sum_data, sum_data, 2);
            const qint8x16_t final_sum = vqaddq_qs8(vqaddq_qs8(sum_data, sum_data2), sum_data3);
            if(pool_stride_x == 2)
            {
                const qint8x8x2_t      table      = { { vget_low_s8(final_sum), vget_high_s8(final_sum) } };
                static const qint8x8_t lookup_val = { 0, 2, 4, 6, 8, 10, 12, 14 };
                res                               = vtbl2_s8(table, lookup_val);
            }
            else
            {
                res = vget_low_s8(final_sum);
            }
            res = vqmul_qs8(res, scale_vec, fixed_point_position);
        }
        else
        {
            const qint8x16_t max_data  = vmaxq_s8(vmaxq_s8(top_data, bottom_data), middle_data);
            const qint8x16_t max_data2 = vextq_s8(max_data, max_data, 1);
            const qint8x16_t max_data3 = vextq_s8(max_data, max_data, 2);
            const qint8x16_t final_max = vmaxq_s8(vmaxq_s8(max_data, max_data2), max_data3);

            if(pool_stride_x == 2)
            {
                const qint8x8x2_t      table      = { { vget_low_s8(final_max), vget_high_s8(final_max) } };
                static const qint8x8_t lookup_val = { 0, 2, 4, 6, 8, 10, 12, 14 };
                res                               = vtbl2_s8(table, lookup_val);
            }
            else
            {
                res = vget_low_s8(final_max);
            }
        }
        vst1_qs8(reinterpret_cast<qint8_t *>(output.ptr()), res);
    },
    input, output);
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling3_f32(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr const int pool_size     = 3;
    int                 pool_pad_x    = 0;
    int                 pool_pad_y    = 0;
    int                 pool_stride_x = 0;
    int                 pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + pool_pad_x;
    const int upper_bound_h = _input->info()->dimension(1) + pool_pad_y;

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const uint8_t *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float32x4_t top_data    = vld1q_f32(reinterpret_cast<const float *>(input_top_ptr + input.offset()));
        const float32x4_t middle_data = vld1q_f32(reinterpret_cast<const float *>(input_middle_ptr + input.offset()));
        const float32x4_t bottom_data = vld1q_f32(reinterpret_cast<const float *>(input_bottom_ptr + input.offset()));
        float32x2_t       res         = {};
        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            float             scale   = calculate_avg_scale(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
            const float32x2_t scale_v = vdup_n_f32(scale);

            // Perform pooling
            const float32x4_t sum_data = vaddq_f32(vaddq_f32(top_data, bottom_data), middle_data);
            res                        = vpadd_f32(vget_high_f32(vsetq_lane_f32(0.f, sum_data, 3)), vget_low_f32(sum_data));
            res                        = vmul_f32(vpadd_f32(res, res), scale_v);
        }
        else
        {
            const float32x4_t max_data = vmaxq_f32(vmaxq_f32(top_data, bottom_data), middle_data);
            res                        = vpmax_f32(vget_high_f32(vsetq_lane_f32(-std::numeric_limits<float>::max(), max_data, 3)), vget_low_f32(max_data));
            res                        = vpmax_f32(res, res);
        }
        *(reinterpret_cast<float *>(output.ptr())) = vget_lane_f32(res, 0);
    },
    input, output);
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling7_f32(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr const int pool_size     = 7;
    int                 pool_pad_x    = 0;
    int                 pool_pad_y    = 0;
    int                 pool_stride_x = 0;
    int                 pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + pool_pad_x;
    const int upper_bound_h = _input->info()->dimension(1) + pool_pad_y;

    std::array<const uint8_t *, pool_size> input_ptrs{ {} };
    for(int i = 0; i < pool_size; ++i)
    {
        input_ptrs[i] = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + i));
    }

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float32x2_t res = {};
        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            float             scale   = calculate_avg_scale(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
            const float32x2_t scale_v = vdup_n_f32(scale);

            // Perform pooling
            float32x4x2_t data     = vld2q_f32(reinterpret_cast<const float *>(input_ptrs[0] + input.offset()));
            float32x4_t   sum_data = vaddq_f32(data.val[0], vsetq_lane_f32(0.f, data.val[1], 3));
            for(int i = 1; i < pool_size; ++i)
            {
                data     = vld2q_f32(reinterpret_cast<const float *>(input_ptrs[i] + input.offset()));
                sum_data = vaddq_f32(sum_data, data.val[0]);
                sum_data = vaddq_f32(sum_data, vsetq_lane_f32(0.f, data.val[1], 3));
            }
            res = vpadd_f32(vget_high_f32(sum_data), vget_low_f32(sum_data));
            res = vmul_f32(vpadd_f32(res, res), scale_v);
        }
        else
        {
            float32x4x2_t max_data = vld2q_f32(reinterpret_cast<const float *>(input_ptrs[0] + input.offset()));
            for(int i = 1; i < pool_size; ++i)
            {
                const float32x4x2_t data = vld2q_f32(reinterpret_cast<const float *>(input_ptrs[i] + input.offset()));
                max_data                 = vmax2q_f32(max_data, data);
            }
            res = vpmax_f32(vget_high_f32(vsetq_lane_f32(-std::numeric_limits<float>::max(), max_data.val[1], 3)), vget_low_f32(max_data.val[1]));
            res = vpmax_f32(res, vpmax_f32(vget_high_f32(max_data.val[0]), vget_low_f32(max_data.val[0])));
            res = vpmax_f32(res, res);
        }
        *(reinterpret_cast<float *>(output.ptr())) = vget_lane_f32(res, 0);
    },
    input, output);
}

void NEPoolingLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    const unsigned int pool_stride_x = _pool_info.pad_stride_info().stride().first;
    const unsigned int pool_stride_y = _pool_info.pad_stride_info().stride().second;

    // Set step for input in x and y direction for the input
    Window       window_input(window);
    unsigned int window_x_inc = 0;
    switch(_input->info()->data_type())
    {
        case DataType::QS8:
        case DataType::F16:
        {
            window_x_inc = (pool_stride_x == 2) ? _num_elems_processed_per_iteration * 2 : _num_elems_processed_per_iteration;
            break;
        }
        case DataType::F32:
        {
            window_x_inc = pool_stride_x;
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported");
        }
    }
    window_input.set(Window::DimX, Window::Dimension(window.x().start() * pool_stride_x, window.x().end() * pool_stride_x, window_x_inc));
    window_input.set(Window::DimY, Window::Dimension(window.y().start() * pool_stride_y, window.y().end() * pool_stride_y, pool_stride_y));

    // Run function
    (this->*_func)(window_input, window);
}
