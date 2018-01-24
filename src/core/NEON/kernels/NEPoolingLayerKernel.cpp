/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <tuple>

using namespace arm_compute;

namespace
{
void auto_init(const ITensorInfo *input, ITensorInfo *output, unsigned int pooled_w, unsigned int pooled_h)
{
    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(0, pooled_w);
    output_shape.set(1, pooled_h);

    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));
}

template <bool exclude_padding>
inline float calculate_avg_scale(const Coordinates &id, const int pool_size, const int upper_bound_w, const int upper_bound_h,
                                 const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int       start_x = id.x() * stride_x - pad_x;
    int       start_y = id.y() * stride_y - pad_y;
    const int end_x   = std::min(start_x + pool_size, upper_bound_w);
    const int end_y   = std::min(start_y + pool_size, upper_bound_h);
    if(exclude_padding)
    {
        start_x = std::max(0, start_x);
        start_y = std::max(0, start_y);
    }
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
    return sshr_qs8(scale_values_q8[val], (7 - fixed_point_position));
}

inline qint16_t calculate_avg_scale_q16(const Coordinates &id, int pool_size, int upper_bound_w, int upper_bound_h,
                                        int pad_x, int pad_y, int stride_x, int stride_y, int fixed_point_position)
{
    static std::array<qint16_t, 10> scale_values_q16 =
    { { 0x0, 0x0, 0x4000, 0x2AAB, 0x2000, 0x199A, 0x1555, 0x1249, 0x1000, 0xE38 } };
    const int start_x = id.x() * stride_x - pad_x;
    const int start_y = id.y() * stride_y - pad_y;
    const int end_x   = std::min(start_x + pool_size, upper_bound_w);
    const int end_y   = std::min(start_y + pool_size, upper_bound_h);
    const int val     = ((end_y - start_y) * (end_x - start_x));
    return sshr_qs16(scale_values_q16[val], (15 - fixed_point_position));
}

template <bool exclude_padding>
inline void scale_vector_s16x8(uint16x8_t &v, const Coordinates &id, int id_offset, int step,
                               const int pool_size, const int upper_bound_w, const int upper_bound_h,
                               const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int       start_x = (id.x() + id_offset) * stride_x - pad_x;
    int       start_y = id.y() * stride_y - pad_y;
    const int end_y   = std::min(start_y + pool_size, upper_bound_h);
    if(exclude_padding)
    {
        start_y = std::max(0, start_y);
    }

    std::array<uint16_t, 8> elems =
    {
        {
            vgetq_lane_u16(v, 0),
            vgetq_lane_u16(v, 1),
            vgetq_lane_u16(v, 2),
            vgetq_lane_u16(v, 3),
            vgetq_lane_u16(v, 4),
            vgetq_lane_u16(v, 5),
            vgetq_lane_u16(v, 6),
            vgetq_lane_u16(v, 7),
        }
    };

    for(auto &el : elems)
    {
        int       c_start_x = start_x;
        const int end_x     = std::min(c_start_x + pool_size, upper_bound_w);
        if(exclude_padding)
        {
            c_start_x = std::max(0, c_start_x);
        }
        float scale = 1.f / ((end_y - start_y) * (end_x - c_start_x));
        el *= scale;
        start_x += step * stride_x;
    }

    v = vsetq_lane_u16(elems[0], v, 0);
    v = vsetq_lane_u16(elems[1], v, 1);
    v = vsetq_lane_u16(elems[2], v, 2);
    v = vsetq_lane_u16(elems[3], v, 3);
    v = vsetq_lane_u16(elems[4], v, 4);
    v = vsetq_lane_u16(elems[5], v, 5);
    v = vsetq_lane_u16(elems[6], v, 6);
    v = vsetq_lane_u16(elems[7], v, 7);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, unsigned int &pooled_w, unsigned int pooled_h, int pool_size)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    int                 pool_pad_x        = 0;
    int                 pool_pad_y        = 0;
    int                 pool_stride_x     = 0;
    int                 pool_stride_y     = 0;
    PoolingType         pool_type         = pool_info.pool_type();
    const PadStrideInfo pad_stride_info   = pool_info.pad_stride_info();
    const bool          exclude_padding   = pool_info.exclude_padding();
    const bool          is_global_pooling = pool_info.is_global_pooling();
    std::tie(pool_pad_x, pool_pad_y)       = pad_stride_info.pad();
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    static const std::set<int> supported_pool_sizes = { 2, 3 };

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(pool_type == PoolingType::L2 && is_data_type_quantized(input->data_type()));
    ARM_COMPUTE_RETURN_ERROR_ON((supported_pool_sizes.find(pool_size) == supported_pool_sizes.end()) && ((input->data_type() != DataType::F32) && (input->data_type() != DataType::QASYMM8)));
    ARM_COMPUTE_RETURN_ERROR_ON(!is_global_pooling && (pool_pad_x >= pool_size || pool_pad_y >= pool_size));
    ARM_COMPUTE_RETURN_ERROR_ON(is_global_pooling && (input->tensor_shape().x() != input->tensor_shape().y()));
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_fixed_point(input->data_type()) && pool_stride_x > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(exclude_padding && is_data_type_fixed_point(input->data_type()));

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON((output->dimension(0) != pooled_w) || (output->dimension(1) != pooled_h));
    }

    return Status{};
}

Status validate_arguments_pool_info(const ITensorInfo *input, const PoolingLayerInfo &pool_info, const unsigned int pool_size)
{
    const bool is_global_pooling = pool_info.is_global_pooling();
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_global_pooling && (input->tensor_shape().x() != input->tensor_shape().y()),
                                    "Global pooling is supported only with rectangular inputs!");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_global_pooling && ((pool_info.pad_stride_info().pad().first >= pool_size) || (pool_info.pad_stride_info().pad().second >= pool_size)),
                                    "Invalid pool size and pool pad combination!");

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &pool_info, unsigned int &num_elems_processed_per_iteration,
                                                        BorderSize &border_size,
                                                        unsigned int pooled_w, unsigned int pooled_h, int pool_size)
{
    unsigned int        num_elems_read_per_iteration = 0;
    unsigned int        num_elems_horizontal_window  = 0;
    int                 pool_pad_x                   = 0;
    int                 pool_pad_y                   = 0;
    int                 pool_stride_x                = 0;
    int                 pool_stride_y                = 0;
    const int           input_width                  = input->dimension(0);
    const int           input_height                 = input->dimension(1);
    const PadStrideInfo pad_stride_info              = pool_info.pad_stride_info();
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    std::tie(pool_pad_x, pool_pad_y)       = pad_stride_info.pad();

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->dimension(0),
                                                     input->dimension(1),
                                                     pool_size,
                                                     pool_size,
                                                     pad_stride_info);

    // Select element size
    switch(input->data_type())
    {
        case DataType::QS8:
            num_elems_read_per_iteration = 16;
            switch(pool_size)
            {
                case 2:
                    num_elems_processed_per_iteration = (pool_stride_x == 2) ? 8 : 15;
                    break;
                case 3:
                    num_elems_processed_per_iteration = (pool_stride_x == 2) ? 7 : 14;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Pooling size not supported");
                    break;
            }
            num_elems_horizontal_window = (pool_stride_x == 2) ? 8 : 16;
            break;
        case DataType::QASYMM8:
            switch(pool_size)
            {
                case 2:
                    num_elems_read_per_iteration      = 16;
                    num_elems_processed_per_iteration = (pool_stride_x == 2) ? 8 : 15;
                    num_elems_horizontal_window       = (pool_stride_x == 2) ? 8 : 16;
                    break;
                case 3:
                    num_elems_read_per_iteration      = 16;
                    num_elems_processed_per_iteration = (pool_stride_x == 2) ? 7 : 14;
                    num_elems_horizontal_window       = (pool_stride_x == 2) ? 8 : 16;
                    break;
                default:
                    num_elems_read_per_iteration      = 1;
                    num_elems_processed_per_iteration = 1;
                    num_elems_horizontal_window       = 1;
                    break;
            }
            break;
        case DataType::QS16:
            num_elems_read_per_iteration = 8;
            switch(pool_size)
            {
                case 2:
                    num_elems_processed_per_iteration = (pool_stride_x == 2) ? 4 : 7;
                    break;
                case 3:
                    num_elems_processed_per_iteration = (pool_stride_x == 2) ? 3 : 6;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Pooling size not supported");
            }
            num_elems_horizontal_window = (pool_stride_x == 2) ? 4 : 8;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
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
                    num_elems_read_per_iteration = 1; // We use vload4 for poolingN but with a leftover for loop
                    break;
            }
            num_elems_processed_per_iteration = 1;
            num_elems_horizontal_window       = 1;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    // Number of iterations in X dimension
    const int num_iterations_x = (pooled_w + num_elems_processed_per_iteration - 1) / num_elems_processed_per_iteration;

    // Upper limit for the number of right/bottom border elements that are accessed
    const int upper_bound_w = ((num_iterations_x - 1) * num_elems_processed_per_iteration * pool_stride_x - pool_pad_x + num_elems_read_per_iteration) - input_width;
    const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

    border_size         = BorderSize(pool_pad_y, pool_pad_x);
    border_size.right   = std::max(upper_bound_w, pool_pad_x);
    border_size.bottom  = std::max(upper_bound_h, pool_pad_y);
    bool window_changed = false;

    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(0, pooled_w);
    output_shape.set(1, pooled_h);
    TensorInfo output_info(input->clone()->set_tensor_shape(output_shape));

    Window             win = calculate_max_window(output_info, Steps(num_elems_processed_per_iteration));
    AccessWindowStatic input_access(input, -pool_pad_x, -pool_pad_y, input_width + border_size.right, input_height + border_size.bottom);

    if(output->total_size() != 0)
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_horizontal_window);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }
    else
    {
        window_changed = update_window_and_padding(win, input_access);
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
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
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    const PoolingType   pool_type         = pool_info.pool_type();
    const PadStrideInfo pad_stride_info   = pool_info.pad_stride_info();
    const bool          exclude_padding   = pool_info.exclude_padding();
    const bool          is_global_pooling = pool_info.is_global_pooling();
    const int           pool_stride_x     = pad_stride_info.stride().first;

    // Update pool size in case of global pooling
    const int pool_size = is_global_pooling ? input->info()->dimension(0) : pool_info.pool_size();

    // Validate pool info before calling scaled_dimensions
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_pool_info(input->info(), pool_info, pool_size));

    // Check output dimensions
    unsigned int pooled_w, pooled_h;
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->info()->dimension(0),
                                                     input->info()->dimension(1),
                                                     pool_size,
                                                     pool_size,
                                                     pad_stride_info);

    // Output auto initialization if not yet initialized
    auto_init(input->info(), output->info(), pooled_w, pooled_h);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), pool_info, pooled_w, pooled_h, pool_size));

    // Set instance variables
    _input     = input;
    _output    = output;
    _pool_info = pool_info;

    // Get data type
    const DataType data_type = input->info()->data_type();

    // Select appropriate function
    if(data_type == DataType::QS8)
    {
        switch(pool_size)
        {
            case 2:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = &NEPoolingLayerKernel::pooling2_q8<PoolingType::AVG>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling2_q8<PoolingType::MAX>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            case 3:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = &NEPoolingLayerKernel::pooling3_q8<PoolingType::AVG>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling3_q8<PoolingType::MAX>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported pooling size!");
        }
    }
    else if(data_type == DataType::QASYMM8)
    {
        if(pool_size == 2 && pool_stride_x < 3)
        {
            switch(pool_type)
            {
                case PoolingType::AVG:
                    _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling2_qasymm8<PoolingType::AVG, true> : &NEPoolingLayerKernel::pooling2_qasymm8<PoolingType::AVG, false>;
                    break;
                case PoolingType::MAX:
                    _func = &NEPoolingLayerKernel::pooling2_qasymm8<PoolingType::MAX>;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported pooling type!");
            }
        }
        else if(pool_size == 3 && pool_stride_x < 3)
        {
            switch(pool_type)
            {
                case PoolingType::AVG:
                    _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling3_qasymm8<PoolingType::AVG, true> : &NEPoolingLayerKernel::pooling3_qasymm8<PoolingType::AVG, false>;
                    break;
                case PoolingType::MAX:
                    _func = &NEPoolingLayerKernel::pooling3_qasymm8<PoolingType::MAX>;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported pooling type!");
            }
        }
        else
        {
            switch(pool_type)
            {
                case PoolingType::AVG:
                    _func = (exclude_padding) ? &NEPoolingLayerKernel::poolingN_qasymm8<PoolingType::AVG, true> : &NEPoolingLayerKernel::poolingN_qasymm8<PoolingType::AVG, false>;
                    break;
                case PoolingType::MAX:
                    _func = &NEPoolingLayerKernel::poolingN_qasymm8<PoolingType::MAX>;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported pooling type!");
            }
        }
    }
    else if(data_type == DataType::QS16)
    {
        switch(pool_size)
        {
            case 2:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = &NEPoolingLayerKernel::pooling2_q16<PoolingType::AVG>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling2_q16<PoolingType::MAX>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            case 3:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = &NEPoolingLayerKernel::pooling3_q16<PoolingType::AVG>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling3_q16<PoolingType::MAX>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported pooling size!");
        }
    }
    else if(data_type == DataType::F16)
    {
        switch(pool_size)
        {
            case 2:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling2_f16<PoolingType::AVG, true> : &NEPoolingLayerKernel::pooling2_f16<PoolingType::AVG, false>;
                        break;
                    case PoolingType::L2:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling2_f16<PoolingType::L2, true> : &NEPoolingLayerKernel::pooling2_f16<PoolingType::L2, false>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling2_f16<PoolingType::MAX, false>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            case 3:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling3_f16<PoolingType::AVG, true> : &NEPoolingLayerKernel::pooling3_f16<PoolingType::AVG, false>;
                        break;
                    case PoolingType::L2:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling3_f16<PoolingType::L2, true> : &NEPoolingLayerKernel::pooling3_f16<PoolingType::L2, false>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling3_f16<PoolingType::MAX, false>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported pooling size!");
        }
    }
    else if(data_type == DataType::F32)
    {
        switch(pool_size)
        {
            case 2:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling2_f32<PoolingType::AVG, true> : &NEPoolingLayerKernel::pooling2_f32<PoolingType::AVG, false>;
                        break;
                    case PoolingType::L2:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling2_f32<PoolingType::L2, true> : &NEPoolingLayerKernel::pooling2_f32<PoolingType::L2, false>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling2_f32<PoolingType::MAX, false>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            case 3:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling3_f32<PoolingType::AVG, true> : &NEPoolingLayerKernel::pooling3_f32<PoolingType::AVG, false>;
                        break;
                    case PoolingType::L2:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling3_f32<PoolingType::L2, true> : &NEPoolingLayerKernel::pooling3_f32<PoolingType::L2, false>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling3_f32<PoolingType::MAX, false>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            case 7:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling7_f32<PoolingType::AVG, true> : &NEPoolingLayerKernel::pooling7_f32<PoolingType::AVG, false>;
                        break;
                    case PoolingType::L2:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::pooling7_f32<PoolingType::L2, true> : &NEPoolingLayerKernel::pooling7_f32<PoolingType::L2, false>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::pooling7_f32<PoolingType::MAX, false>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
            default:
                switch(pool_type)
                {
                    case PoolingType::AVG:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::poolingN_f32<PoolingType::AVG, true> : &NEPoolingLayerKernel::poolingN_f32<PoolingType::AVG, false>;
                        break;
                    case PoolingType::L2:
                        _func = (exclude_padding) ? &NEPoolingLayerKernel::poolingN_f32<PoolingType::L2, true> : &NEPoolingLayerKernel::poolingN_f32<PoolingType::L2, false>;
                        break;
                    case PoolingType::MAX:
                        _func = &NEPoolingLayerKernel::poolingN_f32<PoolingType::MAX, false>;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported pooling type!");
                }
                break;
        }
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), pool_info, _num_elems_processed_per_iteration, _border_size, pooled_w, pooled_h, pool_size);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
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
        qint8x8_t  lower_res   = {};
        qint8x8_t  upper_res   = {};
        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            const qint8_t   scale     = calculate_avg_scale_q8(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y, fixed_point_position);
            const qint8x8_t scale_vec = vdup_n_qs8(scale);

            // Perform pooling
            const qint8x16_t sum_data = vqaddq_qs8(top_data, bottom_data);
            lower_res                 = vqmul_qs8(vpadd_s8(vget_low_s8(sum_data), vget_high_s8(sum_data)), scale_vec, fixed_point_position);
            if(pool_stride_x == 1)
            {
                const qint8x16_t sum_data_shifted = vextq_s8(sum_data, sum_data, 1);
                upper_res                         = vqmul_qs8(vpadd_s8(vget_low_s8(sum_data_shifted), vget_high_s8(sum_data_shifted)), scale_vec, fixed_point_position);
            }
        }
        else
        {
            const qint8x16_t max_data = vmaxq_s8(top_data, bottom_data);
            lower_res                 = vpmax_s8(vget_low_s8(max_data), vget_high_s8(max_data));
            if(pool_stride_x == 1)
            {
                const qint8x16_t max_data_shifted = vextq_s8(max_data, max_data, 1);
                upper_res                         = vpmax_s8(vget_low_s8(max_data_shifted), vget_high_s8(max_data_shifted));
            }
        }
        if(pool_stride_x == 1)
        {
            const qint8x8x2_t res = { { lower_res, upper_res } };
            vst2_s8(reinterpret_cast<qint8_t *>(output.ptr()), res);
        }
        else
        {
            vst1_qs8(reinterpret_cast<qint8_t *>(output.ptr()), lower_res);
        }
    },
    input, output);
}

template <PoolingType pooling_type, bool exclude_padding>
void NEPoolingLayerKernel::pooling2_qasymm8(const Window &window_input, const Window &window)
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
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));

    const int scale_step_x = (pool_stride_x == 1) ? 2 : 1;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = vld1q_u8(reinterpret_cast<const uint8_t *>(input_top_ptr + input.offset()));
        const auto bottom_data = vld1q_u8(reinterpret_cast<const uint8_t *>(input_bottom_ptr + input.offset()));
        uint8x8_t  lower_res   = {};
        uint8x8_t  upper_res   = {};

        if(pooling_type != PoolingType::MAX)
        {
            const uint16x8x2_t top_data_u16    = { { vmovl_u8(vget_low_u8(top_data)), vmovl_u8(vget_high_u8(top_data)) } };
            const uint16x8x2_t bottom_data_u16 = { { vmovl_u8(vget_low_u8(bottom_data)), vmovl_u8(vget_high_u8(bottom_data)) } };

            // Add rows
            const uint16x8x2_t vrsum =
            {
                {
                    vaddq_u16(top_data_u16.val[0], bottom_data_u16.val[0]),
                    vaddq_u16(top_data_u16.val[1], bottom_data_u16.val[1]),
                }
            };

            // Pair-wise add row data
            const uint16x4x2_t vpsum =
            {
                {
                    vpadd_u16(vget_low_u16(vrsum.val[0]), vget_high_u16(vrsum.val[0])),
                    vpadd_u16(vget_low_u16(vrsum.val[1]), vget_high_u16(vrsum.val[1])),
                }
            };

            uint16x8_t res_lower = vcombine_u16(vpsum.val[0], vpsum.val[1]);

            // Scale lower result
            scale_vector_s16x8<exclude_padding>(res_lower, id, 0, scale_step_x,
                                                pool_size, upper_bound_w, upper_bound_h,
                                                pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
            lower_res = vmovn_u16(res_lower);

            // Compute upper result for stride_x == 1
            if(pool_stride_x == 1)
            {
                // Shifted row sum
                const uint16x8x2_t vrsum_shifted =
                {
                    {
                        vextq_u16(vrsum.val[0], vrsum.val[1], 1),
                        vextq_u16(vrsum.val[1], vrsum.val[1], 1)
                    }
                };

                // Pair-wise add shifted row
                const uint16x4x2_t vpsum_shifted =
                {
                    {
                        vpadd_u16(vget_low_u16(vrsum_shifted.val[0]), vget_high_u16(vrsum_shifted.val[0])),
                        vpadd_u16(vget_low_u16(vrsum_shifted.val[1]), vget_high_u16(vrsum_shifted.val[1])),
                    }
                };
                uint16x8_t res_upper = vcombine_u16(vpsum_shifted.val[0], vpsum_shifted.val[1]);

                // Scale lower result
                scale_vector_s16x8<exclude_padding>(res_upper, id, 1, 2,
                                                    pool_size, upper_bound_w, upper_bound_h,
                                                    pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
                upper_res = vmovn_u16(res_upper);
            }
        }
        else
        {
            const uint8x16_t max_data = vmaxq_u8(top_data, bottom_data);
            lower_res                 = vpmax_u8(vget_low_u8(max_data), vget_high_u8(max_data));
            if(pool_stride_x == 1)
            {
                const uint8x16_t max_data_shifted = vextq_u8(max_data, max_data, 1);
                upper_res                         = vpmax_u8(vget_low_u8(max_data_shifted), vget_high_u8(max_data_shifted));
            }
        }

        // Store result
        if(pool_stride_x == 1)
        {
            const uint8x8x2_t res = { { lower_res, upper_res } };
            vst2_u8(reinterpret_cast<uint8_t *>(output.ptr()), res);
        }
        else
        {
            vst1_u8(reinterpret_cast<uint8_t *>(output.ptr()), lower_res);
        }
    },
    input, output);
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling2_q16(const Window &window_input, const Window &window)
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

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = vld1q_qs16(reinterpret_cast<const qint16_t *>(input_top_ptr + input.offset()));
        const auto bottom_data = vld1q_qs16(reinterpret_cast<const qint16_t *>(input_bottom_ptr + input.offset()));
        qint16x4_t lower_res   = {};
        qint16x4_t upper_res   = {};
        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            const qint16_t   scale     = calculate_avg_scale_q16(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y, fixed_point_position);
            const qint16x4_t scale_vec = vdup_n_qs16(scale);

            // Perform pooling
            const qint16x8_t sum_data = vqaddq_qs16(top_data, bottom_data);
            lower_res                 = vqmul_qs16(vpadd_s16(vget_low_s16(sum_data), vget_high_s16(sum_data)), scale_vec, fixed_point_position);
            if(pool_stride_x == 1)
            {
                const qint16x8_t sum_data_shifted = vextq_s16(sum_data, sum_data, 1);
                upper_res                         = vqmul_qs16(vpadd_s16(vget_low_s16(sum_data_shifted), vget_high_s16(sum_data_shifted)), scale_vec, fixed_point_position);
            }
        }
        else
        {
            const qint16x8_t max_data = vmaxq_s16(top_data, bottom_data);
            lower_res                 = vpmax_s16(vget_low_s16(max_data), vget_high_s16(max_data));
            if(pool_stride_x == 1)
            {
                const qint16x8_t max_data_shifted = vextq_s16(max_data, max_data, 1);
                upper_res                         = vpmax_s16(vget_low_s16(max_data_shifted), vget_high_s16(max_data_shifted));
            }
        }
        if(pool_stride_x == 1)
        {
            const qint16x4x2_t res = { { lower_res, upper_res } };
            vst2_s16(reinterpret_cast<qint16_t *>(output.ptr()), res);
        }
        else
        {
            vst1_qs16(reinterpret_cast<qint16_t *>(output.ptr()), lower_res);
        }
    },
    input, output);
}

template <PoolingType pooling_type, bool exclude_padding>
void NEPoolingLayerKernel::pooling3_f16(const Window &window_input, const Window &window)
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr const int pool_size     = 3;
    int                 pool_pad_x    = 0;
    int                 pool_pad_y    = 0;
    int                 pool_stride_x = 0;
    int                 pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const unsigned char *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float16x4_t top_data    = vld1_f16(reinterpret_cast<const float16_t *>(input_top_ptr + input.offset()));
        float16x4_t middle_data = vld1_f16(reinterpret_cast<const float16_t *>(input_middle_ptr + input.offset()));
        float16x4_t bottom_data = vld1_f16(reinterpret_cast<const float16_t *>(input_bottom_ptr + input.offset()));
        float16x4_t res         = {};

        // Get power of 2 in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            top_data    = vmul_f16(top_data, top_data);
            middle_data = vmul_f16(middle_data, middle_data);
            bottom_data = vmul_f16(bottom_data, bottom_data);
        }

        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            const float       scale   = calculate_avg_scale<exclude_padding>(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
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

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            res = vinv_f16(vinvsqrt_f16(res));
        }

        *(reinterpret_cast<float16_t *>(output.ptr())) = vget_lane_f16(res, 0);
    },
    input, output);
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(window_input);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR("FP16 Not supported! Recompile the library with arch=arm64-v8.2-a");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

template <PoolingType pooling_type, bool exclude_padding>
void NEPoolingLayerKernel::pooling2_f16(const Window &window_input, const Window &window)
{
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator      input(_input, window_input);
    Iterator      output(_output, window);
    constexpr int pool_size = 2;
    int           pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto        top_data    = vld2q_f16(reinterpret_cast<const float16_t *>(input_top_ptr + input.offset()));
        auto        bottom_data = vld2q_f16(reinterpret_cast<const float16_t *>(input_bottom_ptr + input.offset()));
        float16x8_t res         = {};

        // Get power of 2 in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            top_data.val[0]    = vmulq_f16(top_data.val[0], top_data.val[0]);
            top_data.val[1]    = vmulq_f16(top_data.val[1], top_data.val[1]);
            bottom_data.val[0] = vmulq_f16(bottom_data.val[0], bottom_data.val[0]);
            bottom_data.val[1] = vmulq_f16(bottom_data.val[1], bottom_data.val[1]);
        }

        if(pooling_type != PoolingType::MAX)
        {
            const float       scale   = calculate_avg_scale<exclude_padding>(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
            const float16x8_t scale_v = vdupq_n_f16(scale);
            res                       = vmulq_f16(scale_v, vaddq_f16(bottom_data.val[1], vaddq_f16(bottom_data.val[0], vaddq_f16(top_data.val[0], top_data.val[1]))));
        }
        else
        {
            res = vmaxq_f16(bottom_data.val[1], vmaxq_f16(bottom_data.val[0], vmaxq_f16(top_data.val[0], top_data.val[1])));
        }

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            res = vinvq_f16(vinvsqrtq_f16(res));
        }

        // Store result
        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), res);
    },
    input, output);
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(window_input);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR("FP16 Not supported! Recompile the library with arch=arm64-v8.2-a");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

template <PoolingType pooling_type, bool exclude_padding>
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
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float32x2_t top_data    = vld1_f32(reinterpret_cast<const float *>(input_top_ptr + input.offset()));
        float32x2_t bottom_data = vld1_f32(reinterpret_cast<const float *>(input_bottom_ptr + input.offset()));
        float32x2_t res         = {};
        float       final_res   = 0;

        // Get power of 2 in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            top_data    = vmul_f32(top_data, top_data);
            bottom_data = vmul_f32(bottom_data, bottom_data);
        }

        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            float             scale   = calculate_avg_scale<exclude_padding>(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
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
        final_res = vget_lane_f32(res, 0);

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            final_res = sqrt(final_res);
        }

        // Store result
        *(reinterpret_cast<float *>(output.ptr())) = final_res;
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
            const qint8_t scale = calculate_avg_scale_q8(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y, fixed_point_position);

            // Perform pooling for stride 2
            const qint8x16_t sum_data  = vqaddq_qs8(vqaddq_qs8(top_data, bottom_data), middle_data);
            const qint8x16_t sum_data2 = vextq_s8(sum_data, sum_data, 1);
            const qint8x16_t sum_data3 = vextq_s8(sum_data, sum_data, 2);
            const qint8x16_t final_sum = vqaddq_qs8(vqaddq_qs8(sum_data, sum_data2), sum_data3);
            if(pool_stride_x == 2)
            {
                const qint8x8x2_t      table      = { { vget_low_s8(final_sum), vget_high_s8(final_sum) } };
                static const qint8x8_t lookup_val = { 0, 2, 4, 6, 8, 10, 12, 14 };
                const qint8x8_t        scale_vec  = vdup_n_qs8(scale);
                res                               = vtbl2_s8(table, lookup_val);
                res                               = vqmul_qs8(res, scale_vec, fixed_point_position);
                vst1_qs8(reinterpret_cast<qint8_t *>(output.ptr()), res);
            }
            else
            {
                const qint8x16_t scale_vec = vdupq_n_qs8(scale);
                vst1q_qs8(reinterpret_cast<qint8_t *>(output.ptr()), vqmulq_qs8(final_sum, scale_vec, fixed_point_position));
            }
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
                vst1_qs8(reinterpret_cast<qint8_t *>(output.ptr()), res);
            }
            else
            {
                vst1q_qs8(reinterpret_cast<qint8_t *>(output.ptr()), final_max);
            }
        }
    },
    input, output);
}

template <PoolingType pooling_type, bool exclude_padding>
void NEPoolingLayerKernel::pooling3_qasymm8(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr int pool_size     = 3;
    int           pool_pad_x    = 0;
    int           pool_pad_y    = 0;
    int           pool_stride_x = 0;
    int           pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const uint8_t *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = vld1q_u8(reinterpret_cast<const uint8_t *>(input_top_ptr + input.offset()));
        const auto middle_data = vld1q_u8(reinterpret_cast<const uint8_t *>(input_middle_ptr + input.offset()));
        const auto bottom_data = vld1q_u8(reinterpret_cast<const uint8_t *>(input_bottom_ptr + input.offset()));

        if(pooling_type == PoolingType::AVG)
        {
            // Convert data to u16
            const uint16x8x2_t top_data_u16    = { { vmovl_u8(vget_low_u8(top_data)), vmovl_u8(vget_high_u8(top_data)) } };
            const uint16x8x2_t middle_data_u16 = { { vmovl_u8(vget_low_u8(middle_data)), vmovl_u8(vget_high_u8(middle_data)) } };
            const uint16x8x2_t bottom_data_u16 = { { vmovl_u8(vget_low_u8(bottom_data)), vmovl_u8(vget_high_u8(bottom_data)) } };

            // Calculate row sums
            const uint16x8x2_t vrsum =
            {
                {
                    vaddq_u16(vaddq_u16(top_data_u16.val[0], bottom_data_u16.val[0]), middle_data_u16.val[0]),
                    vaddq_u16(vaddq_u16(top_data_u16.val[1], bottom_data_u16.val[1]), middle_data_u16.val[1]),
                }
            };
            const uint16x8x2_t vrsum_shifted_1 =
            {
                {
                    vextq_u16(vrsum.val[0], vrsum.val[1], 1),
                    vextq_u16(vrsum.val[1], vrsum.val[1], 1)
                }
            };
            const uint16x8x2_t vrsum_shifted_2 =
            {
                {
                    vextq_u16(vrsum.val[0], vrsum.val[1], 2),
                    vextq_u16(vrsum.val[1], vrsum.val[1], 2)
                }
            };
            // Calculate final sum
            uint16x8x2_t final_sum =
            {
                {
                    vaddq_u16(vaddq_u16(vrsum.val[0], vrsum_shifted_1.val[0]), vrsum_shifted_2.val[0]),
                    vaddq_u16(vaddq_u16(vrsum.val[1], vrsum_shifted_1.val[1]), vrsum_shifted_2.val[1]),
                }
            };
            if(pool_stride_x == 2)
            {
                uint16x8_t res =
                {
                    vgetq_lane_u16(final_sum.val[0], 0),
                    vgetq_lane_u16(final_sum.val[0], 2),
                    vgetq_lane_u16(final_sum.val[0], 4),
                    vgetq_lane_u16(final_sum.val[0], 6),
                    vgetq_lane_u16(final_sum.val[1], 0),
                    vgetq_lane_u16(final_sum.val[1], 2),
                    vgetq_lane_u16(final_sum.val[1], 4),
                    vgetq_lane_u16(final_sum.val[1], 6),
                };

                scale_vector_s16x8<exclude_padding>(res, id, 0, 1,
                                                    pool_size, upper_bound_w, upper_bound_h,
                                                    pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr()), vmovn_u16(res));
            }
            else
            {
                // Scale lower result
                scale_vector_s16x8<exclude_padding>(final_sum.val[0], id, 0, 1,
                                                    pool_size, upper_bound_w, upper_bound_h,
                                                    pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
                // Scale lower result
                scale_vector_s16x8<exclude_padding>(final_sum.val[1], id, 8, 1,
                                                    pool_size, upper_bound_w, upper_bound_h,
                                                    pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
                const uint8x16_t res = vcombine_u8(vmovn_u16(final_sum.val[0]), vmovn_u16(final_sum.val[1]));
                vst1q_u8(reinterpret_cast<uint8_t *>(output.ptr()), res);
            }
        }
        else
        {
            const uint8x16_t max_data        = vmaxq_u8(vmaxq_u8(top_data, bottom_data), middle_data);
            const uint8x16_t max_data_shift1 = vextq_u8(max_data, max_data, 1);
            const uint8x16_t max_data_shift2 = vextq_u8(max_data, max_data, 2);
            const uint8x16_t final_max       = vmaxq_u8(vmaxq_u8(max_data, max_data_shift1), max_data_shift2);

            if(pool_stride_x == 2)
            {
                const uint8x8x2_t      table      = { { vget_low_u8(final_max), vget_high_u8(final_max) } };
                static const uint8x8_t lookup_val = { 0, 2, 4, 6, 8, 10, 12, 14 };
                const uint8x8_t        res        = vtbl2_u8(table, lookup_val);
                vst1_u8(reinterpret_cast<uint8_t *>(output.ptr()), res);
            }
            else
            {
                vst1q_u8(reinterpret_cast<uint8_t *>(output.ptr()), final_max);
            }
        }
    },
    input, output);
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling3_q16(const Window &window_input, const Window &window)
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

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const unsigned char *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = vld1q_qs16(reinterpret_cast<const qint16_t *>(input_top_ptr + input.offset()));
        const auto middle_data = vld1q_qs16(reinterpret_cast<const qint16_t *>(input_middle_ptr + input.offset()));
        const auto bottom_data = vld1q_qs16(reinterpret_cast<const qint16_t *>(input_bottom_ptr + input.offset()));

        if(pooling_type == PoolingType::AVG)
        {
            // Calculate scale
            const qint16_t scale = calculate_avg_scale_q16(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y, fixed_point_position);

            // Perform pooling for stride 2
            const qint16x8_t sum_data  = vqaddq_qs16(vqaddq_qs16(top_data, bottom_data), middle_data);
            const qint16x8_t sum_data2 = vextq_s16(sum_data, sum_data, 1);
            const qint16x8_t sum_data3 = vextq_s16(sum_data, sum_data, 2);
            const qint16x8_t final_sum = vqaddq_qs16(vqaddq_qs16(sum_data, sum_data2), sum_data3);
            if(pool_stride_x == 2)
            {
                const qint16x4_t tmp       = { vgetq_lane_s16(final_sum, 0), vgetq_lane_s16(final_sum, 2), vgetq_lane_s16(final_sum, 4), vgetq_lane_s16(final_sum, 6) };
                const qint16x4_t scale_vec = vdup_n_qs16(scale);
                vst1_qs16(reinterpret_cast<qint16_t *>(output.ptr()), vqmul_qs16(tmp, scale_vec, fixed_point_position));
            }
            else
            {
                const qint16x8_t scale_vec = vdupq_n_qs16(scale);
                vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()), vqmulq_qs16(final_sum, scale_vec, fixed_point_position));
            }
        }
        else
        {
            const qint16x8_t max_data  = vmaxq_s16(vmaxq_s16(top_data, bottom_data), middle_data);
            const qint16x8_t max_data2 = vextq_s16(max_data, max_data, 1);
            const qint16x8_t max_data3 = vextq_s16(max_data, max_data, 2);
            const qint16x8_t final_max = vmaxq_s16(vmaxq_s16(max_data, max_data2), max_data3);

            if(pool_stride_x == 2)
            {
                const qint16x4_t tmp = { vgetq_lane_s16(final_max, 0), vgetq_lane_s16(final_max, 2), vgetq_lane_s16(final_max, 4), vgetq_lane_s16(final_max, 6) };
                vst1_qs16(reinterpret_cast<qint16_t *>(output.ptr()), tmp);
            }
            else
            {
                vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()), final_max);
            }
        }
    },
    input, output);
}

template <PoolingType pooling_type, bool exclude_padding>
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
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const uint8_t *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float32x4_t top_data    = vld1q_f32(reinterpret_cast<const float *>(input_top_ptr + input.offset()));
        float32x4_t middle_data = vld1q_f32(reinterpret_cast<const float *>(input_middle_ptr + input.offset()));
        float32x4_t bottom_data = vld1q_f32(reinterpret_cast<const float *>(input_bottom_ptr + input.offset()));
        float32x2_t res         = {};
        float       final_res   = 0;

        // Get power of 2 in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            top_data    = vmulq_f32(top_data, top_data);
            middle_data = vmulq_f32(middle_data, middle_data);
            bottom_data = vmulq_f32(bottom_data, bottom_data);
        }

        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            float             scale   = calculate_avg_scale<exclude_padding>(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
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
        final_res = vget_lane_f32(res, 0);

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            final_res = sqrt(final_res);
        }

        // Store result
        *(reinterpret_cast<float *>(output.ptr())) = final_res;
    },
    input, output);
}

template <PoolingType pooling_type, bool exclude_padding>
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
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    std::array<const uint8_t *, pool_size> input_ptrs{ {} };
    for(int i = 0; i < pool_size; ++i)
    {
        input_ptrs[i] = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + i));
    }

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float32x2_t res       = {};
        float       final_res = 0.f;
        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            float             scale   = calculate_avg_scale<exclude_padding>(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);
            const float32x2_t scale_v = vdup_n_f32(scale);

            // Perform pooling
            float32x4x2_t data = vld2q_f32(reinterpret_cast<const float *>(input_ptrs[0] + input.offset()));
            // Get power of 2 in case of l2 pooling
            if(pooling_type == PoolingType::L2)
            {
                data.val[0] = vmulq_f32(data.val[0], data.val[0]);
                data.val[1] = vmulq_f32(data.val[1], data.val[1]);
            }
            float32x4_t sum_data = vaddq_f32(data.val[0], vsetq_lane_f32(0.f, data.val[1], 3));
            for(int i = 1; i < pool_size; ++i)
            {
                data = vld2q_f32(reinterpret_cast<const float *>(input_ptrs[i] + input.offset()));
                // Get power of 2 in case of l2 pooling
                if(pooling_type == PoolingType::L2)
                {
                    data.val[0] = vmulq_f32(data.val[0], data.val[0]);
                    data.val[1] = vmulq_f32(data.val[1], data.val[1]);
                }
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
        final_res = vget_lane_f32(res, 0);

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            final_res = sqrt(final_res);
        }

        // Store result
        *(reinterpret_cast<float *>(output.ptr())) = final_res;
    },
    input, output);
}

template <PoolingType pooling_type, bool exclude_padding>
void NEPoolingLayerKernel::poolingN_f32(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    const int pool_size     = _pool_info.is_global_pooling() ? _input->info()->tensor_shape().x() : _pool_info.pool_size();
    int       pool_pad_x    = 0;
    int       pool_pad_y    = 0;
    int       pool_stride_x = 0;
    int       pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float res = 0.0f;

        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            const float scale = calculate_avg_scale<exclude_padding>(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);

            // Perform pooling
            float32x4_t vres = vdupq_n_f32(0.0f);

            for(int y = 0; y < pool_size; ++y)
            {
                int x = 0;
                for(; x <= (pool_size - 4); x += 4)
                {
                    const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_x) * _input->info()->strides_in_bytes().x() +
                                                                                       (y - pool_pad_y) * _input->info()->strides_in_bytes().y()));

                    // Get power of 2 in case of l2 pooling and accumulate
                    if(pooling_type == PoolingType::L2)
                    {
                        vres = vmlaq_f32(vres, data, data);
                    }
                    else
                    {
                        vres = vaddq_f32(vres, data);
                    }
                }

                // Leftover for loop
                for(; x < pool_size; ++x)
                {
                    float data = *(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_x) * _input->info()->strides_in_bytes().x() + (y - pool_pad_y) * _input->info()->strides_in_bytes().y()));

                    // Get power of 2 in case of l2 pooling
                    if(pooling_type == PoolingType::L2)
                    {
                        data *= data;
                    }

                    res += data;
                }
            }

#if defined(__aarch64__)
            // Reduction operation available on 64 bit architectures only
            res += vaddvq_f32(vres);
#else  // __aarch64__
            // Reduction
            float32x2_t tmp = vpadd_f32(vget_high_f32(vres), vget_low_f32(vres));
            tmp             = vpadd_f32(tmp, tmp);

            res += vget_lane_f32(tmp, 0);
#endif // __aarch64__
            // Divide by scale
            res *= scale;
        }
        else
        {
            float32x4_t vres = vdupq_n_f32(std::numeric_limits<float>::min());
            res              = std::numeric_limits<float>::min();

            for(int y = 0; y < pool_size; ++y)
            {
                int x = 0;
                for(; x <= (pool_size - 4); x += 4)
                {
                    const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_x) * _input->info()->strides_in_bytes().x() +
                                                                                       (y - pool_pad_y) * _input->info()->strides_in_bytes().y()));
                    vres                   = vmaxq_f32(vres, data);
                }

                // Leftover for loop
                for(; x < pool_size; ++x)
                {
                    const float data = *(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_x) * _input->info()->strides_in_bytes().x() + (y - pool_pad_y) * _input->info()->strides_in_bytes().y()));
                    res              = std::max(res, data);
                }
            }

#if defined(__aarch64__)
            // Reduction operation available on 64 bit architectures only
            res = std::max(vmaxvq_f32(vres), res);
#else  // __aarch64__
            float32x2_t tmp = vpmax_f32(vget_high_f32(vres), vget_low_f32(vres));
            tmp             = vpmax_f32(tmp, tmp);

            res = std::max(res, vget_lane_f32(tmp, 0));
#endif // __aarch64__
        }

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            res = std::sqrt(res);
        }

        // Store result
        *(reinterpret_cast<float *>(output.ptr())) = res;
    },
    input, output);
}

template <PoolingType pooling_type, bool exclude_padding>
void NEPoolingLayerKernel::poolingN_qasymm8(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    const int pool_size     = _pool_info.is_global_pooling() ? _input->info()->tensor_shape().x() : _pool_info.pool_size();
    int       pool_pad_x    = 0;
    int       pool_pad_y    = 0;
    int       pool_stride_x = 0;
    int       pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_x);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_y);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        uint8_t res = 0;

        if(pooling_type != PoolingType::MAX)
        {
            uint32x4_t vres = vdupq_n_u32(0);
            uint32_t   sres = 0;

            // Calculate scale
            const float scale = calculate_avg_scale<exclude_padding>(id, pool_size, upper_bound_w, upper_bound_h, pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y);

            // Perform pooling
            for(int y = 0; y < pool_size; ++y)
            {
                int x = 0;
                for(; x <= (pool_size - 8); x += 8)
                {
                    const uint8x8_t data = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + (x - pool_pad_x) * _input->info()->strides_in_bytes().x() + (y - pool_pad_y) * _input->info()->strides_in_bytes().y()));

                    const uint16x8_t data_u16 = vmovl_u8(data);
                    vres                      = vaddq_u32(vres, vaddl_u16(vget_high_u16(data_u16), vget_low_u16(data_u16)));
                }

                // Leftover for loop
                for(; x < pool_size; ++x)
                {
                    uint8_t data = *(reinterpret_cast<const uint8_t *>(input.ptr() + (x - pool_pad_x) * _input->info()->strides_in_bytes().x() + (y - pool_pad_y) * _input->info()->strides_in_bytes().y()));
                    sres += data;
                }
            }

            // Reduction
            const auto tmp = vpadd_u32(vget_high_u32(vres), vget_low_u32(vres));
            sres += vget_lane_u32(tmp, 0) + vget_lane_u32(tmp, 1);

            // Divide by scale
            res = static_cast<uint8_t>(support::cpp11::round(sres * scale));
        }
        else
        {
            uint8x8_t vres = vdup_n_u8(0);
            res            = 0;

            for(int y = 0; y < pool_size; ++y)
            {
                int x = 0;
                for(; x <= (pool_size - 8); x += 8)
                {
                    const uint8x8_t data = vld1_u8(reinterpret_cast<const uint8_t *>(input.ptr() + (x - pool_pad_x) * _input->info()->strides_in_bytes().x() + (y - pool_pad_y) * _input->info()->strides_in_bytes().y()));
                    vres                 = vmax_u8(vres, data);
                }

                // Leftover for loop
                for(; x < pool_size; ++x)
                {
                    const uint8_t data = *(reinterpret_cast<const uint8_t *>(input.ptr() + (x - pool_pad_x) * _input->info()->strides_in_bytes().x() + (y - pool_pad_y) * _input->info()->strides_in_bytes().y()));
                    res                = std::max(res, data);
                }
            }

            // Reduce max
            vres = vpmax_u8(vres, vres);
            vres = vpmax_u8(vres, vres);
            vres = vpmax_u8(vres, vres);

            // Get max value
            res = std::max(res, vget_lane_u8(vres, 0));
        }

        // Store result
        *(reinterpret_cast<uint8_t *>(output.ptr())) = res;
    },
    input, output);
}

Status NEPoolingLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);

    unsigned int pooled_w                          = 0;
    unsigned int pooled_h                          = 0;
    unsigned int num_elems_processed_per_iteration = 0;
    BorderSize   border_size(0);

    const bool         is_global_pooling = pool_info.is_global_pooling();
    const unsigned int pool_size         = is_global_pooling ? input->tensor_shape().x() : pool_info.pool_size();

    // Validate pool info befor calling scaled_dimensions
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_pool_info(input, pool_info, pool_size));

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->dimension(0),
                                                     input->dimension(1),
                                                     pool_size,
                                                     pool_size,
                                                     pool_info.pad_stride_info());

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, pool_info, pooled_w, pooled_h, pool_size));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), pool_info, num_elems_processed_per_iteration, border_size, pooled_w, pooled_h, pool_size).first);

    return Status{};
}

void NEPoolingLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    const unsigned int pool_stride_x = _pool_info.pad_stride_info().stride().first;
    const unsigned int pool_stride_y = _pool_info.pad_stride_info().stride().second;
    const unsigned int pool_size     = _pool_info.pool_size();

    // Set step for input in x and y direction for the input
    Window       window_input(window);
    unsigned int window_x_inc = 0;
    switch(_input->info()->data_type())
    {
        case DataType::QS8:
        case DataType::QS16:
        case DataType::F16:
        {
            window_x_inc = (pool_stride_x == 2) ? _num_elems_processed_per_iteration * 2 : _num_elems_processed_per_iteration;
            break;
        }
        case DataType::QASYMM8:
        {
            window_x_inc = pool_stride_x;
            if((pool_size == 2 || pool_size == 3) && pool_stride_x < 3)
            {
                window_x_inc = (pool_stride_x == 2) ? _num_elems_processed_per_iteration * 2 : _num_elems_processed_per_iteration;
            }
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
