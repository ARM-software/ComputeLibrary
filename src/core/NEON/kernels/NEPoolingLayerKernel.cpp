/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <tuple>

namespace arm_compute
{
using namespace misc::shape_calculator;

namespace
{
inline float calculate_avg_scale(bool exclude_padding, DataLayout data_layout, const Coordinates &id, const int pool_size_x, const int pool_size_y, const int upper_bound_w, const int upper_bound_h,
                                 const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    const unsigned int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    int start_x = id[idx_width] * stride_x - pad_x;
    int start_y = id[idx_height] * stride_y - pad_y;

    const int end_x = std::min(start_x + pool_size_x, upper_bound_w);
    const int end_y = std::min(start_y + pool_size_y, upper_bound_h);
    if(exclude_padding)
    {
        start_x = std::max(0, start_x);
        start_y = std::max(0, start_y);
    }
    return 1.f / ((end_y - start_y) * (end_x - start_x));
}

template <typename T, typename TVec>
inline void scale_vector_q16x8(bool exclude_padding, TVec &v, const Coordinates &id, int id_offset, int step,
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

    std::array<T, 8> elems =
    {
        {
            wrapper::vgetlane(v, 0),
            wrapper::vgetlane(v, 1),
            wrapper::vgetlane(v, 2),
            wrapper::vgetlane(v, 3),
            wrapper::vgetlane(v, 4),
            wrapper::vgetlane(v, 5),
            wrapper::vgetlane(v, 6),
            wrapper::vgetlane(v, 7),
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

    v = wrapper::vsetlane(elems[0], v, 0);
    v = wrapper::vsetlane(elems[1], v, 1);
    v = wrapper::vsetlane(elems[2], v, 2);
    v = wrapper::vsetlane(elems[3], v, 3);
    v = wrapper::vsetlane(elems[4], v, 4);
    v = wrapper::vsetlane(elems[5], v, 5);
    v = wrapper::vsetlane(elems[6], v, 6);
    v = wrapper::vsetlane(elems[7], v, 7);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, unsigned int &pooled_w, unsigned int pooled_h)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    PoolingType         pool_type       = pool_info.pool_type;
    const PadStrideInfo pad_stride_info = pool_info.pad_stride_info;
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(pool_type == PoolingType::L2 && is_data_type_quantized(input->data_type()));

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON((output->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH)) != pooled_w)
                                    || (output->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT)) != pooled_h));
    }

    return Status{};
}

Status validate_arguments_pool_info(const unsigned int pool_size_x, const unsigned int pool_size_y)
{
    ARM_COMPUTE_RETURN_ERROR_ON(pool_size_x == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(pool_size_y == 0);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &pool_info, unsigned int &num_elems_processed_per_iteration,
                                                        BorderSize &border_size,
                                                        unsigned int pooled_w, unsigned int pooled_h, int pool_size_x, int pool_size_y)
{
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_pool_shape(*input, pool_info)));

    const auto          data_layout                  = pool_info.data_layout == DataLayout::UNKNOWN ? input->data_layout() : pool_info.data_layout;
    unsigned int        num_elems_read_per_iteration = 0;
    unsigned int        num_elems_horizontal_window  = 0;
    int                 pool_stride_x                = 0;
    int                 pool_stride_y                = 0;
    const int           idx_width                    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int           idx_height                   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int           input_width                  = input->dimension(idx_width);
    const int           input_height                 = input->dimension(idx_height);
    const PadStrideInfo pad_stride_info              = pool_info.pad_stride_info;
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();
    const int  pool_pad_right  = pad_stride_info.pad_right();
    const int  pool_pad_top    = pad_stride_info.pad_top();
    const int  pool_pad_left   = pad_stride_info.pad_left();
    const int  pool_pad_bottom = pad_stride_info.pad_bottom();
    const bool is_square       = pool_size_x == pool_size_y;

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->dimension(idx_width),
                                                     input->dimension(idx_height),
                                                     pool_size_x,
                                                     pool_size_y,
                                                     pad_stride_info);

    //If it's not squared and optimized will be executed the MxN
    num_elems_read_per_iteration      = 1;
    num_elems_processed_per_iteration = 1;
    num_elems_horizontal_window       = 1;

    const bool is_nhwc = data_layout == DataLayout::NHWC;

    if(is_square)
    {
        switch(input->data_type())
        {
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
                if(is_nhwc)
                {
                    num_elems_processed_per_iteration = 16;
                    break;
                }
                switch(pool_size_x)
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
                        break;
                }
                break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
                if(is_nhwc)
                {
                    num_elems_processed_per_iteration = 8;
                    break;
                }
                switch(pool_size_x)
                {
                    case 2:
                    case 3:
                        num_elems_read_per_iteration      = 4;
                        num_elems_processed_per_iteration = 1;
                        num_elems_horizontal_window       = 1;
                        break;
                    default:
                        break;
                }
                break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
            case DataType::F32:
                if(is_nhwc)
                {
                    num_elems_processed_per_iteration = 4;
                    break;
                }
                switch(pool_size_x)
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
                        break;
                }
                num_elems_processed_per_iteration = 1;
                num_elems_horizontal_window       = 1;
                break;
            default:
                ARM_COMPUTE_ERROR("Element size not supported");
                break;
        }
    }
    else
    {
        if(is_nhwc)
        {
            num_elems_processed_per_iteration = 16 / input->element_size();
        }
    }

    bool   window_changed = false;
    Window win{};
    if(data_layout == DataLayout::NCHW)
    {
        // Number of iterations in X dimension
        const int num_iterations_x = (pooled_w + num_elems_processed_per_iteration - 1) / num_elems_processed_per_iteration;

        // Upper limit for the number of right/bottom border elements that are accessed
        const int upper_bound_w = ((num_iterations_x - 1) * num_elems_processed_per_iteration * pool_stride_x - pool_pad_left + num_elems_read_per_iteration) - input_width;
        const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_top + pool_size_y) - input_height;

        border_size        = BorderSize(pool_pad_top, pool_pad_right, pool_pad_bottom, pool_pad_left);
        border_size.right  = std::max(upper_bound_w, pool_pad_right);
        border_size.bottom = std::max(upper_bound_h, pool_pad_bottom);

        TensorShape output_shape{ input->tensor_shape() };
        output_shape.set(0, pooled_w);
        output_shape.set(1, pooled_h);
        TensorInfo output_info(input->clone()->set_tensor_shape(output_shape));

        win = calculate_max_window(output_info, Steps(num_elems_processed_per_iteration));
        AccessWindowStatic input_access(input, -pool_pad_left, -pool_pad_top, input_width + border_size.right, input_height + border_size.bottom);

        AccessWindowHorizontal output_access(output, 0, num_elems_horizontal_window);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }
    else
    {
        TensorShape output_shape{ input->tensor_shape() };
        output_shape.set(1, pooled_w);
        output_shape.set(2, pooled_h);
        TensorInfo output_info(input->clone()->set_tensor_shape(output_shape));

        win = calculate_max_window(output_info, Steps(num_elems_processed_per_iteration));
        AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);

        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = update_window_and_padding(win, input_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

template <typename T>
inline T vcvtq_q32_f32(float32x4_t values);

template <>
inline uint32x4_t vcvtq_q32_f32(float32x4_t values)
{
#ifdef __aarch64__
    return vcvtnq_u32_f32(values);
#else  //__aarch64__
    return vcvtq_u32_f32(values);
#endif //__aarch64__
}

template <>
inline int32x4_t vcvtq_q32_f32(float32x4_t values)
{
#ifdef __aarch64__
    return vcvtnq_s32_f32(values);
#else  //__aarch64__
    return vcvtq_s32_f32(values);
#endif //__aarch64__
}

template <typename T>
inline float32x4_t vcvtq_f32_q32(T values);

template <>
inline float32x4_t vcvtq_f32_q32(uint32x4_t values)
{
    return vcvtq_f32_u32(values);
}

template <>
inline float32x4_t vcvtq_f32_q32(int32x4_t values)
{
    return vcvtq_f32_s32(values);
}

template <typename Tout>
inline Tout vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset);

template <>
inline uint8x16_t vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset)
{
    const float new_scale = quant_rescale / scale_pooling;
    return vquantize(acc, UniformQuantizationInfo(new_scale, new_offset));
}

template <>
inline int8x16_t vrequantize_pooling_with_scale(const float32x4x4_t &acc, const float quant_rescale, const float scale_pooling, const int32_t new_offset)
{
    const float new_scale = quant_rescale / scale_pooling;
    return vquantize_signed(acc, UniformQuantizationInfo(new_scale, new_offset));
}

template <typename Tin, typename Tout>
inline Tout vrequantize_pooling(Tin vec1, Tin vec2, const UniformQuantizationInfo &requant_qinfo);

template <>
inline uint8x16_t vrequantize_pooling(uint8x8_t vec1, uint8x8_t vec2, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x4_t acc =
    {
        {
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec1))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec1))))),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec2))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec2))))),
        }
    };
    return vquantize(acc, requant_qinfo);
}

template <>
inline int8x16_t vrequantize_pooling(int8x8_t vec1, int8x8_t vec2, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x4_t acc =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec1))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec1))))),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec2))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec2))))),
        }
    };
    return vquantize_signed(acc, requant_qinfo);
}

template <typename T>
inline T vrequantize_pooling(T &vec, const UniformQuantizationInfo &requant_qinfo);

template <>
inline uint8x8_t vrequantize_pooling(uint8x8_t &vec, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x2_t acc =
    {
        {
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((vec))))),
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8((vec))))),
        }
    };
    return vquantize(acc, requant_qinfo);
}

template <>
inline int8x8_t vrequantize_pooling(int8x8_t &vec, const UniformQuantizationInfo &requant_qinfo)
{
    const float32x4x2_t acc =
    {
        {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8((vec))))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8((vec))))),
        }
    };
    return vquantize_signed(acc, requant_qinfo);
}

} // namespace

NEPoolingLayerKernel::NEPoolingLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _pool_info(), _data_layout(DataLayout::UNKNOWN), _num_elems_processed_per_iteration(0), _border_size(0), _is_square(false)
{
}

BorderSize NEPoolingLayerKernel::border_size() const
{
    return _border_size;
}

void NEPoolingLayerKernel::configure(const ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    const PadStrideInfo pad_stride_info   = pool_info.pad_stride_info;
    const bool          is_global_pooling = pool_info.is_global_pooling;
    const int           pool_stride_x     = pad_stride_info.stride().first;

    // Get data layout
    const auto data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? input->info()->data_layout() : pool_info.data_layout;
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Update pool size in case of global pooling
    const Size2D pool_size(
        is_global_pooling ? input->info()->dimension(idx_width) : pool_info.pool_size.width,
        is_global_pooling ? input->info()->dimension(idx_height) : pool_info.pool_size.height);

    // Validate pool info before calling scaled_dimensions
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_pool_info(pool_size.x(), pool_size.y()));

    // Check output dimensions
    unsigned int pooled_w;
    unsigned int pooled_h;
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->info()->dimension(idx_width),
                                                     input->info()->dimension(idx_height),
                                                     pool_size.x(),
                                                     pool_size.y(),
                                                     pad_stride_info);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), pool_info, pooled_w, pooled_h));

    // Set instance variables
    _input       = input;
    _output      = output;
    _pool_info   = pool_info;
    _data_layout = input->info()->data_layout();
    _is_square   = (pool_size.x() == pool_size.y());

    // Get data type
    const DataType data_type = input->info()->data_type();
    const bool     is_nchw   = _data_layout == DataLayout::NCHW;

    if(data_type == DataType::QASYMM8)
    {
        if(pool_size.x() == 2 && pool_stride_x < 3 && _is_square)
        {
            if(is_nchw)
            {
                _func = &NEPoolingLayerKernel::pooling2_q8_nchw<uint8_t>;
            }
            else
            {
                _func = &NEPoolingLayerKernel::poolingMxN_q8_nhwc<uint8_t>;
            }
        }
        else if(pool_size.x() == 3 && pool_stride_x < 3 && _is_square)
        {
            if(is_nchw)
            {
                _func = &NEPoolingLayerKernel::pooling3_q8_nchw<uint8_t>;
            }
            else
            {
                _func = &NEPoolingLayerKernel::poolingMxN_q8_nhwc<uint8_t>;
            }
        }
        else
        {
            if(is_nchw)
            {
                _func = &NEPoolingLayerKernel::poolingMxN_q8_nchw<uint8_t>;
            }
            else
            {
                _func = &NEPoolingLayerKernel::poolingMxN_q8_nhwc<uint8_t>;
            }
        }
    }
    else if(data_type == DataType::QASYMM8_SIGNED)
    {
        if(pool_size.x() == 2 && pool_stride_x < 3 && _is_square)
        {
            if(is_nchw)
            {
                _func = &NEPoolingLayerKernel::pooling2_q8_nchw<int8_t>;
            }
            else
            {
                _func = &NEPoolingLayerKernel::poolingMxN_q8_nhwc<int8_t>;
            }
        }
        else if(pool_size.x() == 3 && pool_stride_x < 3 && _is_square)
        {
            if(is_nchw)
            {
                _func = &NEPoolingLayerKernel::pooling3_q8_nchw<int8_t>;
            }
            else
            {
                _func = &NEPoolingLayerKernel::poolingMxN_q8_nhwc<int8_t>;
            }
        }
        else
        {
            if(is_nchw)
            {
                _func = &NEPoolingLayerKernel::poolingMxN_q8_nchw<int8_t>;
            }
            else
            {
                _func = &NEPoolingLayerKernel::poolingMxN_q8_nhwc<int8_t>;
            }
        }
    }
    else if(data_type == DataType::F16)
    {
        if(_is_square)
        {
            switch(pool_size.x())
            {
                case 2:
                {
                    if(is_nchw)
                    {
                        _func = &NEPoolingLayerKernel::pooling2_f16_nchw;
                    }
                    else
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f16_nhwc;
                    }
                }
                break;
                case 3:
                {
                    if(is_nchw)
                    {
                        _func = &NEPoolingLayerKernel::pooling3_f16_nchw;
                    }
                    else
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f16_nhwc;
                    }
                }
                break;
                default:
                {
                    if(is_nchw)
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f16_nchw;
                    }
                    else
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f16_nhwc;
                    }
                    break;
                }
                break;
            }
        }
        else
        {
            if(is_nchw)
            {
                _func = &NEPoolingLayerKernel::poolingMxN_f16_nchw;
            }
            else
            {
                _func = &NEPoolingLayerKernel::poolingMxN_f16_nhwc;
            }
        }
    }
    else if(data_type == DataType::F32)
    {
        if(_is_square)
        {
            switch(pool_size.x())
            {
                case 2:
                {
                    if(is_nchw)
                    {
                        _func = &NEPoolingLayerKernel::pooling2_f32_nchw;
                    }
                    else
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f32_nhwc;
                    }
                    break;
                }
                case 3:
                {
                    if(is_nchw)
                    {
                        _func = &NEPoolingLayerKernel::pooling3_f32_nchw;
                    }
                    else
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f32_nhwc;
                    }
                    break;
                }
                case 7:
                {
                    if(is_nchw)
                    {
                        _func = &NEPoolingLayerKernel::pooling7_f32_nchw;
                    }
                    else
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f32_nhwc;
                    }
                    break;
                }
                default:
                {
                    if(is_nchw)
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f32_nchw;
                    }
                    else
                    {
                        _func = &NEPoolingLayerKernel::poolingMxN_f32_nhwc;
                    }
                    break;
                }
            }
        }
        else
        {
            if(is_nchw)
            {
                _func = &NEPoolingLayerKernel::poolingMxN_f32_nchw;
            }
            else
            {
                _func = &NEPoolingLayerKernel::poolingMxN_f32_nhwc;
            }
        }
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), pool_info, _num_elems_processed_per_iteration, _border_size, pooled_w, pooled_h, pool_size.x(), pool_size.y());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

template <typename T>
void NEPoolingLayerKernel::pooling2_q8_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    /** NEON vector types */
    using q8x8_t    = typename wrapper::traits::neon_vector<T, 8>::type;
    using q8x16_t   = typename wrapper::traits::neon_vector<T, 16>::type;
    using q8x8x2_t  = typename std::conditional<std::is_same<T, uint8_t>::value, uint8x8x2_t, int8x8x2_t>::type;
    using q16_t     = typename wrapper::traits::promote_t<T>;
    using q16x4_t   = typename wrapper::traits::neon_vector<q16_t, 4>::type;
    using q16x8_t   = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q16x8x2_t = typename wrapper::traits::neon_vector<q16_t, 16>::type;

    constexpr int pool_size       = 2;
    int           pool_stride_x   = 0;
    int           pool_stride_y   = 0;
    const int     pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int     pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int     pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int     pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    const T *const input_top_ptr    = reinterpret_cast<const T *>(_input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top))));
    const T *const input_bottom_ptr = reinterpret_cast<const T *>(_input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1)));

    const int scale_step_x = (pool_stride_x == 1) ? 2 : 1;

    const UniformQuantizationInfo input_qinfo          = _input->info()->quantization_info().uniform();
    const UniformQuantizationInfo output_qinfo         = _output->info()->quantization_info().uniform();
    const bool                    have_different_qinfo = input_qinfo != output_qinfo;

    const float                   requant_scale  = output_qinfo.scale / input_qinfo.scale;
    const int32_t                 requant_offset = output_qinfo.offset - static_cast<int32_t>(static_cast<float>(input_qinfo.offset) / requant_scale);
    const UniformQuantizationInfo requant_qinfo  = UniformQuantizationInfo(requant_scale, requant_offset);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = wrapper::vloadq(input_top_ptr + input.offset());
        const auto bottom_data = wrapper::vloadq(input_bottom_ptr + input.offset());
        q8x8_t     lower_res   = {};
        q8x8_t     upper_res   = {};

        if(pooling_type != PoolingType::MAX)
        {
            const q16x8x2_t top_data_q16    = { { wrapper::vmovl(wrapper::vgetlow(top_data)), wrapper::vmovl(wrapper::vgethigh(top_data)) } };
            const q16x8x2_t bottom_data_q16 = { { wrapper::vmovl(wrapper::vgetlow(bottom_data)), wrapper::vmovl(wrapper::vgethigh(bottom_data)) } };

            // Add rows
            const q16x8x2_t vrsum =
            {
                {
                    wrapper::vadd(top_data_q16.val[0], bottom_data_q16.val[0]),
                    wrapper::vadd(top_data_q16.val[1], bottom_data_q16.val[1]),
                }
            };

            // Pair-wise add row data
            const q16x4_t vpsum_1 = wrapper::vpadd(wrapper::vgetlow(vrsum.val[0]), wrapper::vgethigh(vrsum.val[0]));
            const q16x4_t vpsum_2 = wrapper::vpadd(wrapper::vgetlow(vrsum.val[1]), wrapper::vgethigh(vrsum.val[1]));

            q16x8_t res_lower = wrapper::vcombine(vpsum_1, vpsum_2);

            // Scale lower result
            scale_vector_q16x8<q16_t, q16x8_t>(exclude_padding, res_lower, id, 0, scale_step_x,
                                               pool_size, upper_bound_w, upper_bound_h,
                                               pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
            lower_res = wrapper::vmovn(res_lower);

            // Compute upper result for stride_x == 1
            if(pool_stride_x == 1)
            {
                // Shifted row sum
                const q16x8x2_t vrsum_shifted =
                {
                    {
                        wrapper::vext_1(vrsum.val[0], vrsum.val[1]),
                        wrapper::vext_1(vrsum.val[1], vrsum.val[1])
                    }
                };

                // Pair-wise add shifted row
                q16x8_t res_upper = wrapper::vcombine(
                                        wrapper::vpadd(wrapper::vgetlow(vrsum_shifted.val[0]), wrapper::vgethigh(vrsum_shifted.val[0])),
                                        wrapper::vpadd(wrapper::vgetlow(vrsum_shifted.val[1]), wrapper::vgethigh(vrsum_shifted.val[1])));

                // Scale upper result
                scale_vector_q16x8<q16_t, q16x8_t>(exclude_padding, res_upper, id, 1, 2,
                                                   pool_size, upper_bound_w, upper_bound_h,
                                                   pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
                upper_res = wrapper::vmovn(res_upper);
            }
        }
        else
        {
            const q8x16_t max_data = wrapper::vmax(top_data, bottom_data);
            lower_res              = wrapper::vpmax(wrapper::vgetlow(max_data), wrapper::vgethigh(max_data));
            if(pool_stride_x == 1)
            {
                const q8x16_t max_data_shifted = wrapper::vext_1(max_data, max_data);
                upper_res                      = wrapper::vpmax(wrapper::vgetlow(max_data_shifted), wrapper::vgethigh(max_data_shifted));
            }
        }

        if(have_different_qinfo)
        {
            const auto requantized_output = vrequantize_pooling<q8x8_t, q8x16_t>(lower_res, upper_res, requant_qinfo);
            lower_res                     = wrapper::vgetlow(requantized_output);
            upper_res                     = wrapper::vgethigh(requantized_output);
        }

        // Store result
        if(pool_stride_x == 1)
        {
            const q8x8x2_t res = { { lower_res, upper_res } };
            wrapper::vstore(reinterpret_cast<T *>(output.ptr()), res);
        }
        else
        {
            wrapper::vstore(reinterpret_cast<T *>(output.ptr()), lower_res);
        }
    },
    input, output);
}

void NEPoolingLayerKernel::pooling3_f16_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    ARM_COMPUTE_UNUSED(pooling_type);
    ARM_COMPUTE_UNUSED(exclude_padding);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr const int pool_size       = 3;
    const int           pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int           pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int           pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int           pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const unsigned char *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 2));

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
            const float       scale   = calculate_avg_scale(exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
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

void NEPoolingLayerKernel::pooling2_f16_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    ARM_COMPUTE_UNUSED(pooling_type);
    ARM_COMPUTE_UNUSED(exclude_padding);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator      input(_input, window_input);
    Iterator      output(_output, window);
    constexpr int pool_size       = 2;
    const int     pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int     pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int     pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int     pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int           pool_stride_x, pool_stride_y = 0;
    std::tie(pool_stride_x, pool_stride_y)     = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float16x4_t top_data    = vld1_f16(reinterpret_cast<const float16_t *>(input_top_ptr + input.offset()));
        float16x4_t bottom_data = vld1_f16(reinterpret_cast<const float16_t *>(input_bottom_ptr + input.offset()));
        float16x4_t res         = {};

        // Get power of 2 in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            top_data    = vmul_f16(top_data, top_data);
            bottom_data = vmul_f16(bottom_data, bottom_data);
        }

        if(pooling_type != PoolingType::MAX)
        {
            const float       scale   = calculate_avg_scale(exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
            const float16x4_t scale_v = vdup_n_f16(scale);

            const float16x4_t sum_data = vadd_f16(top_data, bottom_data);
            res                        = vmul_f16(vpadd_f16(sum_data, sum_data), scale_v);
        }
        else
        {
            const float16x4_t max_data = vmax_f16(top_data, bottom_data);
            res                        = vpmax_f16(max_data, max_data);
        }

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            res = vinv_f16(vinvsqrt_f16(res));
        }

        // Store result
        *(reinterpret_cast<float16_t *>(output.ptr())) = vget_lane_f16(res, 0);
    },
    input, output);
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(window_input);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR("FP16 Not supported! Recompile the library with arch=arm64-v8.2-a");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

template <typename T>
void NEPoolingLayerKernel::pooling3_q8_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    /** NEON vector types */
    using q8x8_t    = typename wrapper::traits::neon_vector<T, 8>::type;
    using q8x16_t   = typename wrapper::traits::neon_vector<T, 16>::type;
    using q8x8x2_t  = typename std::conditional<std::is_same<T, uint8_t>::value, uint8x8x2_t, int8x8x2_t>::type;
    using q16_t     = typename wrapper::traits::promote_t<T>;
    using q16x8_t   = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q16x8x2_t = typename wrapper::traits::neon_vector<q16_t, 16>::type;

    constexpr int pool_size       = 3;
    const int     pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int     pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int     pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int     pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int           pool_stride_x   = 0;
    int           pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    const UniformQuantizationInfo &input_qinfo  = _input->info()->quantization_info().uniform();
    const UniformQuantizationInfo &output_qinfo = _output->info()->quantization_info().uniform();

    const float                   requant_scale  = output_qinfo.scale / input_qinfo.scale;
    const int32_t                 requant_offset = output_qinfo.offset - static_cast<int32_t>(static_cast<float>(input_qinfo.offset) / requant_scale);
    const UniformQuantizationInfo requant_qinfo  = UniformQuantizationInfo(requant_scale, requant_offset);

    const T *const input_top_ptr    = reinterpret_cast<const T *>(_input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top))));
    const T *const input_middle_ptr = reinterpret_cast<const T *>(_input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1)));
    const T *const input_bottom_ptr = reinterpret_cast<const T *>(_input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 2)));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto top_data    = wrapper::vloadq(input_top_ptr + input.offset());
        const auto middle_data = wrapper::vloadq(input_middle_ptr + input.offset());
        const auto bottom_data = wrapper::vloadq(input_bottom_ptr + input.offset());
        q8x8_t     fres        = {};
        q8x16_t    fqres       = {};

        if(pooling_type == PoolingType::AVG)
        {
            // Convert data to u16
            const q16x8x2_t top_data_q16    = { { wrapper::vmovl(wrapper::vgetlow(top_data)), wrapper::vmovl(wrapper::vgethigh(top_data)) } };
            const q16x8x2_t middle_data_q16 = { { wrapper::vmovl(wrapper::vgetlow(middle_data)), wrapper::vmovl(wrapper::vgethigh(middle_data)) } };
            const q16x8x2_t bottom_data_q16 = { { wrapper::vmovl(wrapper::vgetlow(bottom_data)), wrapper::vmovl(wrapper::vgethigh(bottom_data)) } };

            // Calculate row sums
            const q16x8x2_t vrsum =
            {
                {
                    wrapper::vadd(wrapper::vadd(top_data_q16.val[0], bottom_data_q16.val[0]), middle_data_q16.val[0]),
                    wrapper::vadd(wrapper::vadd(top_data_q16.val[1], bottom_data_q16.val[1]), middle_data_q16.val[1]),
                }
            };
            const q16x8x2_t vrsum_shifted_1 =
            {
                {
                    wrapper::vext_1(vrsum.val[0], vrsum.val[1]),
                    wrapper::vext_1(vrsum.val[1], vrsum.val[1])
                }
            };
            const q16x8x2_t vrsum_shifted_2 =
            {
                {
                    wrapper::vext_2(vrsum.val[0], vrsum.val[1]),
                    wrapper::vext_2(vrsum.val[1], vrsum.val[1])
                }
            };
            // Calculate final sum
            q16x8x2_t final_sum =
            {
                {
                    wrapper::vadd(wrapper::vadd(vrsum.val[0], vrsum_shifted_1.val[0]), vrsum_shifted_2.val[0]),
                    wrapper::vadd(wrapper::vadd(vrsum.val[1], vrsum_shifted_1.val[1]), vrsum_shifted_2.val[1]),
                }
            };
            if(pool_stride_x == 2)
            {
                q16x8_t res =
                {
                    wrapper::vgetlane(final_sum.val[0], 0),
                    wrapper::vgetlane(final_sum.val[0], 2),
                    wrapper::vgetlane(final_sum.val[0], 4),
                    wrapper::vgetlane(final_sum.val[0], 6),
                    wrapper::vgetlane(final_sum.val[1], 0),
                    wrapper::vgetlane(final_sum.val[1], 2),
                    wrapper::vgetlane(final_sum.val[1], 4),
                    wrapper::vgetlane(final_sum.val[1], 6),
                };

                scale_vector_q16x8<q16_t, q16x8_t>(exclude_padding, res, id, 0, 1,
                                                   pool_size, upper_bound_w, upper_bound_h,
                                                   pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
                fres = wrapper::vmovn(res);
            }
            else
            {
                // Scale lower result
                scale_vector_q16x8<q16_t, q16x8_t>(exclude_padding, final_sum.val[0], id, 0, 1,
                                                   pool_size, upper_bound_w, upper_bound_h,
                                                   pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
                // Scale lower result
                scale_vector_q16x8<q16_t, q16x8_t>(exclude_padding, final_sum.val[1], id, 8, 1,
                                                   pool_size, upper_bound_w, upper_bound_h,
                                                   pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
                fqres = wrapper::vcombine(wrapper::vmovn(final_sum.val[0]), wrapper::vmovn(final_sum.val[1]));
            }
        }
        else
        {
            const q8x16_t max_data        = wrapper::vmax(wrapper::vmax(top_data, bottom_data), middle_data);
            const q8x16_t max_data_shift1 = wrapper::vext_1(max_data, max_data);
            const q8x16_t max_data_shift2 = wrapper::vext_2(max_data, max_data);
            const q8x16_t final_max       = wrapper::vmax(wrapper::vmax(max_data, max_data_shift1), max_data_shift2);

            if(pool_stride_x == 2)
            {
                const q8x8x2_t      table      = { { wrapper::vgetlow(final_max), wrapper::vgethigh(final_max) } };
                static const q8x8_t lookup_val = { 0, 2, 4, 6, 8, 10, 12, 14 };
                fres                           = wrapper::vtbl(table, lookup_val);
            }
            else
            {
                fqres = final_max;
            }
        }

        // Store result
        if(pool_stride_x == 1)
        {
            if(input_qinfo != output_qinfo)
            {
                fqres = vrequantize_pooling<q8x8_t, q8x16_t>(wrapper::vgetlow(fqres), wrapper::vgethigh(fqres), requant_qinfo);
            }
            wrapper::vstore(reinterpret_cast<T *>(output.ptr()), fqres);
        }
        else
        {
            if(input_qinfo != output_qinfo)
            {
                fres = vrequantize_pooling<q8x8_t>(fres, requant_qinfo);
            }
            wrapper::vstore(reinterpret_cast<T *>(output.ptr()), fres);
        }
    },
    input, output);
}

void NEPoolingLayerKernel::poolingMxN_f16_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    ARM_COMPUTE_UNUSED(pooling_type);
    ARM_COMPUTE_UNUSED(exclude_padding);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    const int pool_size_x     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().x() : _pool_info.pool_size.width;
    const int pool_size_y     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().y() : _pool_info.pool_size.height;
    const int pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int       pool_stride_x   = 0;
    int       pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float16_t   res  = 0.0f;
        float16x8_t vres = vdupq_n_f16(0.0f);

        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            const float scale = calculate_avg_scale(exclude_padding, DataLayout::NCHW, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);

            // Perform pooling

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 8); x += 8)
                {
                    const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) +
                                                                                           (y - pool_pad_top) * static_cast<int>(_input->info()->strides_in_bytes().y())));

                    // Get power of 2 in case of l2 pooling and accumulate
                    if(pooling_type == PoolingType::L2)
                    {
                        vres = vaddq_f16(vres, vmulq_f16(data, data));
                    }
                    else
                    {
                        vres = vaddq_f16(vres, data);
                    }
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    float16_t data = *(reinterpret_cast<const float16_t *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x())
                                                                           + (y - pool_pad_top) * static_cast<int>(_input->info()->strides_in_bytes().y())));

                    // Get power of 2 in case of l2 pooling
                    if(pooling_type == PoolingType::L2)
                    {
                        data *= data;
                    }

                    res += data;
                }
            }

            // Reduction
            float16x4_t tmp = vpadd_f16(vget_high_f16(vres), vget_low_f16(vres));
            res += vget_lane_f16(tmp, 0);
            res += vget_lane_f16(tmp, 1);
            res += vget_lane_f16(tmp, 2);
            res += vget_lane_f16(tmp, 3);

            // Divide by scale
            res *= scale;
        }
        else
        {
            float16x8_t vres = vdupq_n_f16(std::numeric_limits<float>::lowest());
            res              = std::numeric_limits<float>::lowest();

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 8); x += 8)
                {
                    const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) +
                                                                                           (y - pool_pad_top) * static_cast<int>(_input->info()->strides_in_bytes().y())));
                    vres                   = vmaxq_f16(vres, data);
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    const float16_t data = *(reinterpret_cast<const float16_t *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x())
                                                                                 + (y - pool_pad_top) * static_cast<int>(_input->info()->strides_in_bytes().y())));
                    res = std::max(res, data);
                }
            }

            float16x4_t tmp = vpmax_f16(vget_high_f16(vres), vget_low_f16(vres));
            res             = std::max(res, vget_lane_f16(tmp, 0));
            res             = std::max(res, vget_lane_f16(tmp, 1));
            res             = std::max(res, vget_lane_f16(tmp, 2));
            res             = std::max(res, vget_lane_f16(tmp, 3));
        }

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            res = std::sqrt(res);
        }

        // Store result
        *(reinterpret_cast<float16_t *>(output.ptr())) = res;
    },
    input, output);

#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(window_input);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR("FP16 Not supported! Recompile the library with arch=arm64-v8.2-a");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

void NEPoolingLayerKernel::poolingMxN_f16_nhwc(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    ARM_COMPUTE_UNUSED(pooling_type);
    ARM_COMPUTE_UNUSED(exclude_padding);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    const int pool_size_x     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().y() : _pool_info.pool_size.width;
    const int pool_size_y     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().z() : _pool_info.pool_size.height;
    const int pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int       pool_stride_x   = 0;
    int       pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(2) + (exclude_padding ? 0 : pool_pad_bottom);

    float16x8_t vres;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int idx_width    = id.y() * pool_stride_x;
        const int idx_height   = id.z() * pool_stride_y;
        const int pool_limit_y = pool_pad_top - idx_height;
        const int pool_limit_x = pool_pad_left - idx_width;

        const int pool_start_y = std::max(0, window_input.z().start() + pool_limit_y);
        const int pool_end_y   = std::min(pool_size_y, window_input.z().end() + pool_limit_y);
        const int pool_start_x = std::max(0, window_input.y().start() + pool_limit_x);
        const int pool_end_x   = std::min(pool_size_x, window_input.y().end() + pool_limit_x);

        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            const float scale = calculate_avg_scale(exclude_padding, DataLayout::NHWC, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                    pool_stride_y);
            const float16x8_t scale_v = vdupq_n_f16(scale);

            // Perform pooling
            vres = vdupq_n_f16(0.0f);
            for(int y = pool_start_y; y < pool_end_y; ++y)
            {
                for(int x = pool_start_x; x < pool_end_x; ++x)
                {
                    const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().y()) +
                                                                                           (y - pool_pad_top) * static_cast<int>(_input->info()->strides_in_bytes().z())));

                    // Get power of 2 in case of l2 pooling and accumulate
                    if(pooling_type == PoolingType::L2)
                    {
                        vres = vaddq_f16(vres, vmulq_f16(data, data));
                    }
                    else
                    {
                        vres = vaddq_f16(vres, data);
                    }
                }
            }
            // Divide by scale
            vres = vmulq_f16(vres, scale_v);
        }
        else
        {
            vres = vdupq_n_f16(std::numeric_limits<float>::lowest());

            for(int y = pool_start_y; y < pool_end_y; ++y)
            {
                for(int x = pool_start_x; x < pool_end_x; ++x)
                {
                    const float16x8_t data = vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().y()) +
                                                                                           (y - pool_pad_top) * static_cast<int>(_input->info()->strides_in_bytes().z())));
                    vres                   = vmaxq_f16(vres, data);
                }
            }
        }

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            float16x8_t sqrt_reciprocal = vrsqrteq_f16(vres);
            vres                        = vmulq_f16(vres, vmulq_f16(vrsqrtsq_f16(vmulq_f16(vres, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal));
        }

        // Store result
        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), vres);
    },
    input, output);

#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    ARM_COMPUTE_UNUSED(window_input);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_ERROR("FP16 Not supported! Recompile the library with arch=arm64-v8.2-a");
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

void NEPoolingLayerKernel::poolingMxN_f32_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    const int pool_size_x     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().x() : _pool_info.pool_size.width;
    const int pool_size_y     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().y() : _pool_info.pool_size.height;
    const int pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int       pool_stride_x   = 0;
    int       pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float res = 0.0f;

        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            const float scale = calculate_avg_scale(exclude_padding, DataLayout::NCHW, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);

            // Perform pooling
            float32x4_t vres = vdupq_n_f32(0.0f);

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 4); x += 4)
                {
                    const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                       (_input->info()->strides_in_bytes().y())));

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
                for(; x < pool_size_x; ++x)
                {
                    float data = *(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                   (_input->info()->strides_in_bytes().y())));

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
            float32x4_t vres = vdupq_n_f32(std::numeric_limits<float>::lowest());
            res              = std::numeric_limits<float>::lowest();

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 4); x += 4)
                {
                    const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                       (_input->info()->strides_in_bytes().y())));
                    vres                   = vmaxq_f32(vres, data);
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    const float data = *(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                         (_input->info()->strides_in_bytes().y())));
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

void NEPoolingLayerKernel::pooling2_f32_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr int pool_size       = 2;
    const int     pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int     pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int     pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int     pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int           pool_stride_x   = 0;
    int           pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));

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
            float             scale   = calculate_avg_scale(exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
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

void NEPoolingLayerKernel::pooling3_f32_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr const int pool_size       = 3;
    const int           pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int           pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int           pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int           pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    const uint8_t *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top)));
    const uint8_t *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 1));
    const uint8_t *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + 2));

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
            float             scale   = calculate_avg_scale(exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
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

void NEPoolingLayerKernel::pooling7_f32_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr const int pool_size       = 7;
    const int           pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int           pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int           pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int           pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int                 pool_stride_x   = 0;
    int                 pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    std::array<const uint8_t *, pool_size> input_ptrs{ {} };
    for(int i = 0; i < pool_size; ++i)
    {
        input_ptrs[i] = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_left), -static_cast<int>(pool_pad_top) + i));
    }

    execute_window_loop(window, [&](const Coordinates & id)
    {
        float32x2_t res       = {};
        float       final_res = 0.f;
        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            float             scale   = calculate_avg_scale(exclude_padding, DataLayout::NCHW, id, pool_size, pool_size, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);
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

void NEPoolingLayerKernel::poolingMxN_f32_nhwc(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    const int pool_size_x     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().y() : _pool_info.pool_size.width;
    const int pool_size_y     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().z() : _pool_info.pool_size.height;
    const int pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int       pool_stride_x   = 0;
    int       pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(2) + (exclude_padding ? 0 : pool_pad_bottom);

    float32x4_t vres;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int idx_width    = id.y() * pool_stride_x;
        const int idx_height   = id.z() * pool_stride_y;
        const int pool_limit_y = pool_pad_top - idx_height;
        const int pool_limit_x = pool_pad_left - idx_width;

        const int pool_start_y = std::max(0, window_input.z().start() + pool_limit_y);
        const int pool_end_y   = std::min(pool_size_y, window_input.z().end() + pool_limit_y);
        const int pool_start_x = std::max(0, window_input.y().start() + pool_limit_x);
        const int pool_end_x   = std::min(pool_size_x, window_input.y().end() + pool_limit_x);

        if(pooling_type != PoolingType::MAX)
        {
            // Calculate scale
            const float scale = calculate_avg_scale(exclude_padding, DataLayout::NHWC, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                    pool_stride_y);
            const float32x4_t scale_v = vdupq_n_f32(scale);

            // Perform pooling
            vres = vdupq_n_f32(0.0f);

            for(int y = pool_start_y; y < pool_end_y; ++y)
            {
                for(int x = pool_start_x; x < pool_end_x; ++x)
                {
                    const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                       (_input->info()->strides_in_bytes().z())));

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
            }
            // Divide by scale
            vres = vmulq_f32(vres, scale_v);
        }
        else
        {
            vres = vdupq_n_f32(std::numeric_limits<float>::lowest());
            for(int y = pool_start_y; y < pool_end_y; ++y)
            {
                for(int x = pool_start_x; x < pool_end_x; ++x)
                {
                    const float32x4_t data = vld1q_f32(reinterpret_cast<const float *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                       (_input->info()->strides_in_bytes().z())));
                    vres                   = vmaxq_f32(vres, data);
                }
            }
        }

        // Calculate square-root in case of l2 pooling
        if(pooling_type == PoolingType::L2)
        {
            float32x4_t l2_res = { static_cast<float>(sqrt(vgetq_lane_f32(vres, 0))),
                                   static_cast<float>(sqrt(vgetq_lane_f32(vres, 1))),
                                   static_cast<float>(sqrt(vgetq_lane_f32(vres, 2))),
                                   static_cast<float>(sqrt(vgetq_lane_f32(vres, 3)))
                                 };
            vres = l2_res;
        }

        // Store result
        vst1q_f32(reinterpret_cast<float *>(output.ptr()), vres);
    },
    input, output);
}

template <typename T>
void NEPoolingLayerKernel::poolingMxN_q8_nchw(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    /** NEON vector types */
    using q8x8_t  = typename wrapper::traits::neon_vector<T, 8>::type;
    using q16_t   = typename wrapper::traits::promote_t<T>;
    using q16x8_t = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q32_t   = typename wrapper::traits::promote_t<q16_t>;
    using q32x4_t = typename wrapper::traits::neon_vector<q32_t, 4>::type;

    const int pool_size_x     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().x() : _pool_info.pool_size.width;
    const int pool_size_y     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().y() : _pool_info.pool_size.height;
    const int pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();
    int       pool_stride_x   = 0;
    int       pool_stride_y   = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(0) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_bottom);

    const UniformQuantizationInfo &input_qinfo  = _input->info()->quantization_info().uniform();
    const UniformQuantizationInfo &output_qinfo = _output->info()->quantization_info().uniform();

    execute_window_loop(window, [&](const Coordinates & id)
    {
        T res = std::numeric_limits<T>::min();

        if(pooling_type != PoolingType::MAX)
        {
            q32x4_t vres = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
            q32_t   sres = 0;

            // Calculate scale
            const float scale = calculate_avg_scale(exclude_padding, DataLayout::NCHW, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x, pool_stride_y);

            // Perform pooling
            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 8); x += 8)
                {
                    const q8x8_t data = wrapper::vload(reinterpret_cast<const T *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                   (_input->info()->strides_in_bytes().y())));

                    const q16x8_t data_q16 = wrapper::vmovl(data);
                    vres                   = wrapper::vadd(vres, wrapper::vaddl(wrapper::vgethigh(data_q16), wrapper::vgetlow(data_q16)));
                }

                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    T data = *(reinterpret_cast<const T *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                           (_input->info()->strides_in_bytes().y())));
                    sres += data;
                }
            }

            // Reduction
            const auto tmp = wrapper::vpadd(wrapper::vgethigh(vres), wrapper::vgetlow(vres));
            sres += wrapper::vgetlane(tmp, 0) + wrapper::vgetlane(tmp, 1);

            // Divide by scale
            res = static_cast<T>(support::cpp11::round(sres * scale));
        }
        else
        {
            q8x8_t vres = wrapper::vdup_n(std::numeric_limits<T>::min(), wrapper::traits::vector_64_tag{});

            for(int y = 0; y < pool_size_y; ++y)
            {
                int x = 0;
                for(; x <= (pool_size_x - 8); x += 8)
                {
                    const q8x8_t data = wrapper::vload(reinterpret_cast<const T *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                                   (_input->info()->strides_in_bytes().y())));
                    vres              = wrapper::vmax(vres, data);
                }
                // Leftover for loop
                for(; x < pool_size_x; ++x)
                {
                    const T data = *(reinterpret_cast<const T *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().x()) + (y - pool_pad_top) * static_cast<int>
                                                                 (_input->info()->strides_in_bytes().y())));
                    res          = std::max(res, data);
                }
            }

            // Reduce max
            vres = wrapper::vpmax(vres, vres);
            vres = wrapper::vpmax(vres, vres);
            vres = wrapper::vpmax(vres, vres);

            // Get max value
            res = std::max(res, wrapper::vgetlane(vres, 0));
        }
        // Store result
        res                                    = (input_qinfo != output_qinfo) ? Qasymm8QuantizationHelper<T>::quantize(Qasymm8QuantizationHelper<T>::dequantize(res, input_qinfo), output_qinfo) : res;
        *(reinterpret_cast<T *>(output.ptr())) = res;
    },
    input, output);
}

template <typename T>
void NEPoolingLayerKernel::poolingMxN_q8_nhwc(const Window &window_input, const Window &window, PoolingType pooling_type, bool exclude_padding)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    using q8x8_t  = typename wrapper::traits::neon_vector<T, 8>::type;
    using q8x16_t = typename wrapper::traits::neon_vector<T, 16>::type;
    using q16_t   = typename wrapper::traits::promote_t<T>;
    using q16x8_t = typename wrapper::traits::neon_vector<q16_t, 8>::type;
    using q32_t   = typename wrapper::traits::promote_t<q16_t>;
    using q32x4_t = typename wrapper::traits::neon_vector<q32_t, 4>::type;

    const int pool_size_x     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().y() : _pool_info.pool_size.width;
    const int pool_size_y     = _pool_info.is_global_pooling ? _input->info()->tensor_shape().z() : _pool_info.pool_size.height;
    const int pool_pad_right  = _pool_info.pad_stride_info.pad_right();
    const int pool_pad_top    = _pool_info.pad_stride_info.pad_top();
    const int pool_pad_left   = _pool_info.pad_stride_info.pad_left();
    const int pool_pad_bottom = _pool_info.pad_stride_info.pad_bottom();

    int pool_stride_x = 0;
    int pool_stride_y = 0;
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info.stride();
    const int upper_bound_w = _input->info()->dimension(1) + (exclude_padding ? 0 : pool_pad_right);
    const int upper_bound_h = _input->info()->dimension(2) + (exclude_padding ? 0 : pool_pad_bottom);

    const float32x4_t             half_scale_v = vdupq_n_f32(0.5f);
    const UniformQuantizationInfo input_qinfo  = _input->info()->quantization_info().uniform();
    const UniformQuantizationInfo output_qinfo = _output->info()->quantization_info().uniform();

    const float   quant_rescale = output_qinfo.scale / input_qinfo.scale;
    // "new_offset" doesn't have to consider the "half_scale_v" in its computation
    // With a requantization performed in a single step there won't be uncertainties introduced
    const int32_t new_offset    = output_qinfo.offset - static_cast<int32_t>( static_cast<float>(input_qinfo.offset) / quant_rescale);

    const float                   requant_scale  = output_qinfo.scale / input_qinfo.scale;
    const int32_t                 requant_offset = output_qinfo.offset - static_cast<int32_t>(static_cast<float>(input_qinfo.offset) / requant_scale);
    const UniformQuantizationInfo requant_qinfo  = UniformQuantizationInfo(requant_scale, requant_offset);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int idx_width    = id.y() * pool_stride_x;
        const int idx_height   = id.z() * pool_stride_y;
        const int pool_limit_y = pool_pad_top - idx_height;
        const int pool_limit_x = pool_pad_left - idx_width;

        const int pool_start_y = std::max(0, window_input.z().start() + pool_limit_y);
        const int pool_end_y   = std::min(pool_size_y, window_input.z().end() + pool_limit_y);
        const int pool_start_x = std::max(0, window_input.y().start() + pool_limit_x);
        const int pool_end_x   = std::min(pool_size_x, window_input.y().end() + pool_limit_x);

        if(pooling_type != PoolingType::MAX)
        {
            q32x4_t vres1 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
            q32x4_t vres2 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
            q32x4_t vres3 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});
            q32x4_t vres4 = wrapper::vdup_n(static_cast<q32_t>(0.f), wrapper::traits::vector_128_tag{});

            // Calculate scale
            const float scale = calculate_avg_scale(exclude_padding, DataLayout::NHWC, id, pool_size_x, pool_size_y, upper_bound_w, upper_bound_h, pool_pad_left, pool_pad_top, pool_stride_x,
                                                    pool_stride_y);

            // Perform pooling
            for(int y = pool_start_y; y < pool_end_y; ++y)
            {
                for(int x = pool_start_x; x < pool_end_x; ++x)
                {
                    const q8x16_t data = wrapper::vloadq(reinterpret_cast<const T *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                     (_input->info()->strides_in_bytes().z())));

                    const q16x8_t data_q16  = wrapper::vmovl(wrapper::vgetlow(data));
                    const q16x8_t data2_q16 = wrapper::vmovl(wrapper::vgethigh(data));
                    vres1                   = wrapper::vadd(vres1, wrapper::vmovl(wrapper::vgetlow(data_q16)));
                    vres2                   = wrapper::vadd(vres2, wrapper::vmovl(wrapper::vgethigh(data_q16)));
                    vres3                   = wrapper::vadd(vres3, wrapper::vmovl(wrapper::vgetlow(data2_q16)));
                    vres4                   = wrapper::vadd(vres4, wrapper::vmovl(wrapper::vgethigh(data2_q16)));
                }
            }

            if(input_qinfo != output_qinfo)
            {
                const float32x4x4_t vres =
                {
                    {
                        vcvtq_f32_q32(vres1),
                        vcvtq_f32_q32(vres2),
                        vcvtq_f32_q32(vres3),
                        vcvtq_f32_q32(vres4),
                    }
                };
                const auto requantized_output = vrequantize_pooling_with_scale<q8x16_t>(vres, quant_rescale, scale, new_offset);
                // Store result
                wrapper::vstore(reinterpret_cast<T *>(output.ptr()), wrapper::vgetlow(requantized_output));
                wrapper::vstore(reinterpret_cast<T *>(output.ptr()) + 8, wrapper::vgethigh(requantized_output));
            }
            else
            {
                const float32x4_t scale_v = vdupq_n_f32(scale);
                // Divide by scale and add 0.5f to round to nearest instead of rounding towards zero
                vres1 = vcvtq_q32_f32<q32x4_t>(wrapper::vmla(half_scale_v, vcvtq_f32_q32(vres1), scale_v));
                vres2 = vcvtq_q32_f32<q32x4_t>(wrapper::vmla(half_scale_v, vcvtq_f32_q32(vres2), scale_v));
                vres3 = vcvtq_q32_f32<q32x4_t>(wrapper::vmla(half_scale_v, vcvtq_f32_q32(vres3), scale_v));
                vres4 = vcvtq_q32_f32<q32x4_t>(wrapper::vmla(half_scale_v, vcvtq_f32_q32(vres4), scale_v));

                const q8x8_t res1 = wrapper::vmovn(wrapper::vcombine(wrapper::vmovn(vres1), wrapper::vmovn(vres2)));
                const q8x8_t res2 = wrapper::vmovn(wrapper::vcombine(wrapper::vmovn(vres3), wrapper::vmovn(vres4)));
                // Store result
                wrapper::vstore(reinterpret_cast<T *>(output.ptr()), res1);
                wrapper::vstore(reinterpret_cast<T *>(output.ptr()) + 8, res2);
            }
        }
        else
        {
            q8x16_t vres = wrapper::vdup_n(std::numeric_limits<T>::min(), wrapper::traits::vector_128_tag{});

            for(int y = pool_start_y; y < pool_end_y; ++y)
            {
                for(int x = pool_start_x; x < pool_end_x; ++x)
                {
                    const q8x16_t data = wrapper::vloadq(reinterpret_cast<const T *>(input.ptr() + (x - pool_pad_left) * static_cast<int>(_input->info()->strides_in_bytes().y()) + (y - pool_pad_top) * static_cast<int>
                                                                                     (_input->info()->strides_in_bytes().z())));
                    vres               = wrapper::vmax(vres, data);
                }
            }

            // Store result
            wrapper::vstore(reinterpret_cast<T *>(output.ptr()), (input_qinfo != output_qinfo) ? vrequantize_pooling<q8x8_t, q8x16_t>(wrapper::vgetlow(vres), wrapper::vgethigh(vres), requant_qinfo) : vres);
        }
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

    const bool   is_global_pooling = pool_info.is_global_pooling;
    unsigned int pool_size_x       = 0;
    unsigned int pool_size_y       = 0;

    // Get data layout
    const auto data_layout = pool_info.data_layout == DataLayout::UNKNOWN ? input->data_layout() : pool_info.data_layout;
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    pool_size_x = is_global_pooling ? input->dimension(idx_width) : pool_info.pool_size.width;
    pool_size_y = is_global_pooling ? input->dimension(idx_height) : pool_info.pool_size.height;

    // Validate pool info before calling scaled_dimensions
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_pool_info(pool_size_x, pool_size_y));

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->dimension(idx_width),
                                                     input->dimension(idx_height),
                                                     pool_size_x,
                                                     pool_size_y,
                                                     pool_info.pad_stride_info);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, pool_info, pooled_w, pooled_h));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), pool_info, num_elems_processed_per_iteration, border_size, pooled_w, pooled_h,
                                                              pool_size_x, pool_size_y)
                                .first);

    return Status{};
}

void NEPoolingLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    const unsigned int pool_stride_x   = _pool_info.pad_stride_info.stride().first;
    const unsigned int pool_stride_y   = _pool_info.pad_stride_info.stride().second;
    const unsigned int pool_size       = _pool_info.pool_size.width;
    const bool         exclude_padding = _pool_info.exclude_padding;

    Window window_input(window);
    if(_data_layout == DataLayout::NCHW)
    {
        // Set step for input in x and y direction for the input
        unsigned int window_x_inc = 0;
        switch(_input->info()->data_type())
        {
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
            {
                window_x_inc = pool_stride_x;
                if((pool_size == 2 || pool_size == 3) && pool_stride_x < 3)
                {
                    window_x_inc = (pool_stride_x == 2) ? _num_elems_processed_per_iteration * 2 : _num_elems_processed_per_iteration;
                }
                break;
            }

            case DataType::F16:
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
    }
    else
    {
        window_input.set(Window::DimX, Window::Dimension(window.x().start(), window.x().end(), _num_elems_processed_per_iteration));
        window_input.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), pool_stride_x));
        window_input.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(2), pool_stride_y));
    }

    // Run function
    (this->*_func)(window_input, window, _pool_info.pool_type, exclude_padding);
}
} // namespace arm_compute