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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <algorithm>
#include <arm_neon.h>
#include <limits>
#include <string>
#include <tuple>

using namespace arm_compute;

namespace
{
inline float calculate_avg_scale(const Coordinates &id, const int pool_size, const int upper_bound_w, const int upper_bound_h,
                                 const int pad_x, const int pad_y, const int stride_x, const int stride_y)
{
    int start_x = id.x() * stride_x - pad_x;
    int start_y = id.y() * stride_y - pad_y;
    int end_x   = std::min(start_x + pool_size, upper_bound_w);
    int end_y   = std::min(start_y + pool_size, upper_bound_h);
    return 1.f / ((end_y - start_y) * (end_x - start_x));
}
} // namespace

NEPoolingLayerKernel::NEPoolingLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _pool_info(), _border_size(0)
{
}

BorderSize NEPoolingLayerKernel::border_size() const
{
    return _border_size;
}

void NEPoolingLayerKernel::configure(const ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info)
{
    int                   pool_pad_x      = 0;
    int                   pool_pad_y      = 0;
    int                   pool_stride_x   = 0;
    int                   pool_stride_y   = 0;
    unsigned int          pooled_w        = 0;
    unsigned int          pooled_h        = 0;
    PoolingType           pool_type       = pool_info.pool_type();
    int                   pool_size       = pool_info.pool_size();
    const PadStrideInfo   pad_stride_info = pool_info.pad_stride_info();
    DimensionRoundingType pool_round      = pad_stride_info.round();
    std::tie(pool_pad_x, pool_pad_y)       = pad_stride_info.pad();
    std::tie(pool_stride_x, pool_stride_y) = pad_stride_info.stride();

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON(2 != pool_size && 3 != pool_size);
    ARM_COMPUTE_ERROR_ON(pool_pad_x >= pool_size || pool_pad_y >= pool_size);

    // Check output dimensions
    std::tie(pooled_w, pooled_h) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1),
                                                     pool_size, pool_stride_x, pool_stride_y,
                                                     pool_pad_x, pool_pad_y, pool_round);
    ARM_COMPUTE_UNUSED(pooled_w);
    ARM_COMPUTE_UNUSED(pooled_h);
    ARM_COMPUTE_ERROR_ON((output->info()->dimension(0) != pooled_w) || (output->info()->dimension(1) != pooled_h));

    const int read_elements = (pool_size == 2) ? 2 : 4; // We use vload4 for pooling3
    const int input_width   = input->info()->dimension(0);
    const int input_height  = input->info()->dimension(1);
    const int upper_bound_w = ((pooled_w - 1) * pool_stride_x - pool_pad_x + read_elements) - input_width;
    const int upper_bound_h = ((pooled_h - 1) * pool_stride_y - pool_pad_y + pool_size) - input_height;

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
            _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling2<PoolingType::AVG> : &NEPoolingLayerKernel::pooling2<PoolingType::MAX>;
            break;
        case 3:
            _func = (PoolingType::AVG == pool_type) ? &NEPoolingLayerKernel::pooling3<PoolingType::AVG> : &NEPoolingLayerKernel::pooling3<PoolingType::MAX>;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported pooling size");
            break;
    }

    // Configure kernel window
    const unsigned int     processed_elements = 1;
    Window                 win                = calculate_max_window(*output->info(), Steps(processed_elements));
    AccessWindowStatic     input_access(input->info(), -pool_pad_x, -pool_pad_y, input_width + _border_size.right, input_height + _border_size.bottom);
    AccessWindowHorizontal output_access(output->info(), 0, processed_elements);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));
    INEKernel::configure(win);
}

template <PoolingType pooling_type>
void NEPoolingLayerKernel::pooling2(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

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
void NEPoolingLayerKernel::pooling3(const Window &window_input, const Window &window)
{
    Iterator input(_input, window_input);
    Iterator output(_output, window);

    constexpr const int pool_size = 3;
    int                 pool_pad_x, pool_pad_y, pool_stride_x, pool_stride_y = 0;
    std::tie(pool_pad_x, pool_pad_y)       = _pool_info.pad_stride_info().pad();
    std::tie(pool_stride_x, pool_stride_y) = _pool_info.pad_stride_info().stride();
    const int upper_bound_w = _input->info()->dimension(0) + pool_pad_x;
    const int upper_bound_h = _input->info()->dimension(1) + pool_pad_y;

    const unsigned char *const input_top_ptr    = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y)));
    const unsigned char *const input_middle_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 1));
    const unsigned char *const input_bottom_ptr = _input->ptr_to_element(Coordinates(-static_cast<int>(pool_pad_x), -static_cast<int>(pool_pad_y) + 2));

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

void NEPoolingLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    unsigned int pool_stride_x, pool_stride_y = 0;
    std::tie(pool_stride_x, pool_stride_y)    = _pool_info.pad_stride_info().stride();

    // Set step for input in x and y direction for the input
    Window window_input(window);
    window_input.set(Window::DimX, Window::Dimension(window.x().start() * pool_stride_x, window.x().end() * pool_stride_x, pool_stride_x));
    window_input.set(Window::DimY, Window::Dimension(window.y().start() * pool_stride_y, window.y().end() * pool_stride_y, pool_stride_y));

    // Run function
    (this->*_func)(window_input, window);
}
