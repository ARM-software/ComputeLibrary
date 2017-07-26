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
#include "arm_compute/core/NEON/kernels/NENormalizationLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

NENormalizationLayerKernel::NENormalizationLayerKernel()
    : _func(nullptr), _input(nullptr), _input_squared(nullptr), _output(nullptr), _norm_info(NormType::IN_MAP_1D), _border_size()
{
}

BorderSize NENormalizationLayerKernel::border_size() const
{
    return _border_size;
}

void NENormalizationLayerKernel::configure(const ITensor *input, const ITensor *input_squared, ITensor *output, NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, input_squared, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, input_squared, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, input_squared, output);
    ARM_COMPUTE_ERROR_ON_MSG(!(norm_info.norm_size() % 2), "Normalization size should be odd");
    ARM_COMPUTE_ERROR_ON_VALUE_NOT_REPRESENTABLE_IN_FIXED_POINT(norm_info.beta(), input);
    ARM_COMPUTE_ERROR_ON_VALUE_NOT_REPRESENTABLE_IN_FIXED_POINT(norm_info.kappa(), input);
    ARM_COMPUTE_ERROR_ON_VALUE_NOT_REPRESENTABLE_IN_FIXED_POINT(norm_info.scale_coeff(), input);

    const unsigned int border_width = (norm_info.type() == NormType::CROSS_MAP) ? 0 : std::min(norm_info.norm_size() / 2, 3U);

    _input         = input;
    _input_squared = input_squared;
    _output        = output;
    _norm_info     = norm_info;
    _border_size   = BorderSize(0, border_width);

    unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();

    switch(_input->info()->data_type())
    {
        case DataType::F32:
        {
            num_elems_processed_per_iteration = 4;
            switch(norm_info.type())
            {
                case NormType::IN_MAP_1D:
                    _func = &NENormalizationLayerKernel::normalize_float<DataType::F32, 0, false>;
                    break;
                case NormType::IN_MAP_2D:
                    // Normalize over X and Y
                    _func = &NENormalizationLayerKernel::normalize_float<DataType::F32, 0, true>;
                    break;
                case NormType::CROSS_MAP:
                    _func = &NENormalizationLayerKernel::normalize_float<DataType::F32, 2, false>;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case DataType::F16:
        {
            num_elems_processed_per_iteration = 8;
            switch(norm_info.type())
            {
                case NormType::IN_MAP_1D:
                    _func = &NENormalizationLayerKernel::normalize_float<DataType::F16, 0, false>;
                    break;
                case NormType::IN_MAP_2D:
                    // Normalize over X and Y
                    _func = &NENormalizationLayerKernel::normalize_float<DataType::F16, 0, true>;
                    break;
                case NormType::CROSS_MAP:
                    _func = &NENormalizationLayerKernel::normalize_float<DataType::F16, 2, false>;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case DataType::QS8:
        {
            num_elems_processed_per_iteration = 16;
            switch(norm_info.type())
            {
                case NormType::IN_MAP_1D:
                    _func = &NENormalizationLayerKernel::normalize_fixed_point<DataType::QS8, 0, false>;
                    break;
                case NormType::IN_MAP_2D:
                    // Normalize over X and Y
                    _func = &NENormalizationLayerKernel::normalize_fixed_point<DataType::QS8, 0, true>;
                    break;
                case NormType::CROSS_MAP:
                    _func = &NENormalizationLayerKernel::normalize_fixed_point<DataType::QS8, 2, false>;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        case DataType::QS16:
        {
            num_elems_processed_per_iteration = 8;
            switch(norm_info.type())
            {
                case NormType::IN_MAP_1D:
                    _func = &NENormalizationLayerKernel::normalize_fixed_point<DataType::QS16, 0, false>;
                    break;
                case NormType::IN_MAP_2D:
                    // Normalize over X and Y
                    _func = &NENormalizationLayerKernel::normalize_fixed_point<DataType::QS16, 0, true>;
                    break;
                case NormType::CROSS_MAP:
                    _func = &NENormalizationLayerKernel::normalize_fixed_point<DataType::QS16, 2, false>;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported");
                    break;
            }
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

    const unsigned int num_elems_read_per_iteration = num_elems_processed_per_iteration + 2 * (norm_info.norm_size() / 2);
    const unsigned int num_rows                     = (norm_info.type() == NormType::IN_MAP_2D) ? norm_info.norm_size() : 1;

    // Configure window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowRectangle  input_access(input->info(), -_border_size.left, 0, num_elems_read_per_iteration, num_rows);
    AccessWindowRectangle  input_squared_access(input_squared->info(), -_border_size.left, 0, num_elems_read_per_iteration, num_rows);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, input_squared_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

template <DataType dt, unsigned int dim, bool do_2D_norm>
void NENormalizationLayerKernel::normalize_float(const Window &window)
{
    Iterator input(_input, window);
    Iterator input_squared(_input_squared, window);
    Iterator output(_output, window);

    const int dim_y                = 1;
    const int radius               = _norm_info.norm_size() / 2;
    const int total_size           = _input->info()->dimension(dim) - 1;
    const int input_squared_stride = _input_squared->info()->strides_in_bytes()[dim];
    // We account padding across X only and we iterate over rows
    const int min_left   = (dim == 2) ? 0 : -static_cast<int>(border_size().left);
    const int max_right  = (dim == 2) ? total_size : total_size + border_size().left;
    const int min_top    = 0;
    const int max_bottom = _input->info()->dimension(dim_y) - 1;

    if(dt == DataType::F32)
    {
        const float32x4_t coeff_vec = vdupq_n_f32(_norm_info.scale_coeff());
        const float32x4_t beta_vec  = vdupq_n_f32(_norm_info.beta());
        const float32x4_t kappa_vec = vdupq_n_f32(_norm_info.kappa());

        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Get range to normalize
            const int current_row   = do_2D_norm ? id[dim_y] : 0;
            const int current_slice = id[dim];
            const int first_row     = do_2D_norm ? std::max(current_row - radius, min_top) : 0;
            const int last_row      = do_2D_norm ? std::min(current_row + radius, max_bottom) : 0;
            const int first_slice   = std::max(current_slice - radius, min_left);
            const int last_slice    = std::min(current_slice + radius, max_right);

            // Accumulate 2D In-Map values
            float32x4_t accu = vdupq_n_f32(0.f);
            for(int j = first_row; j <= last_row; j++)
            {
                // Compute row displacement
                const int            row               = (j - current_row) * _input_squared->info()->strides_in_bytes()[dim_y];
                const uint8_t *const input_squared_ptr = input_squared.ptr() + row - (current_slice * input_squared_stride);
                for(int i = first_slice; i <= last_slice; ++i)
                {
                    accu = vaddq_f32(accu, vld1q_f32(reinterpret_cast<const float *>(input_squared_ptr + i * input_squared_stride)));
                }
            }

            // Normalize
            const float32x4_t normalized       = vpowq_f32(vmlaq_f32(kappa_vec, coeff_vec, accu), beta_vec);
            const float32x4_t normalized_pixel = vmulq_f32(vld1q_f32(reinterpret_cast<const float *>(input.ptr())), vinvq_f32(normalized));
            vst1q_f32(reinterpret_cast<float *>(output.ptr()), normalized_pixel);
        },
        input, input_squared, output);
    }
#ifdef ARM_COMPUTE_ENABLE_FP16
    else if(dt == DataType::F16)
    {
        const float16x8_t coeff_vec    = vdupq_n_f16(_norm_info.scale_coeff());
        const float16x8_t beta_vec_f16 = vdupq_n_f16(_norm_info.beta());
        const float16x8_t kappa_vec    = vdupq_n_f16(_norm_info.kappa());

        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Get range to normalize
            const int current_row   = do_2D_norm ? id[dim_y] : 0;
            const int current_slice = id[dim];
            const int first_row     = do_2D_norm ? std::max(current_row - radius, min_top) : 0;
            const int last_row      = do_2D_norm ? std::min(current_row + radius, max_bottom) : 0;
            const int first_slice   = std::max(current_slice - radius, min_left);
            const int last_slice    = std::min(current_slice + radius, max_right);

            // Accumulate 2D In-Map values
            float16x8_t accu = vdupq_n_f16(0.f);
            for(int j = first_row; j <= last_row; j++)
            {
                // Compute row displacement
                const int            row               = (j - current_row) * _input_squared->info()->strides_in_bytes()[dim_y];
                const uint8_t *const input_squared_ptr = input_squared.ptr() + row - (current_slice * input_squared_stride);
                for(int i = first_slice; i <= last_slice; ++i)
                {
                    accu = vaddq_f16(accu, vld1q_f16(reinterpret_cast<const float16_t *>(input_squared_ptr + i * input_squared_stride)));
                }
            }

            const float16x8_t norm_f16         = vpowq_f16(vaddq_f16(kappa_vec, vmulq_f16(coeff_vec, accu)), beta_vec_f16);
            const float16x8_t normalized_pixel = vmulq_f16(vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr())), vinvq_f16(norm_f16));
            vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), normalized_pixel);
        },
        input, input_squared, output);
    }
#endif /* ARM_COMPUTE_ENABLE_FP16 */
    else
    {
        ARM_COMPUTE_ERROR("Not supported");
    }
}

template <DataType dt, unsigned int dim, bool do_2D_norm>
void NENormalizationLayerKernel::normalize_fixed_point(const Window &window)
{
    Iterator input(_input, window);
    Iterator input_squared(_input_squared, window);
    Iterator output(_output, window);

    const int dim_y                = 1;
    const int radius               = _norm_info.norm_size() / 2;
    const int total_size           = _input->info()->dimension(dim) - 1;
    const int input_squared_stride = _input_squared->info()->strides_in_bytes()[dim];
    // We account padding across X only and we iterate over rows
    const int min_left   = (dim == 2) ? 0 : -static_cast<int>(border_size().left);
    const int max_right  = (dim == 2) ? total_size : total_size + border_size().left;
    const int min_top    = 0;
    const int max_bottom = _input->info()->dimension(dim_y) - 1;

    const int fixed_point_position = _input->info()->fixed_point_position();

    if(dt == DataType::QS8)
    {
        const qint8x16_t coeff_vec = vdupq_n_qs8_f32(_norm_info.scale_coeff(), fixed_point_position);
        const qint8x16_t beta_vec  = vdupq_n_qs8_f32(_norm_info.beta(), fixed_point_position);
        const qint8x16_t kappa_vec = vdupq_n_qs8_f32(_norm_info.kappa(), fixed_point_position);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Get range to normalize
            const int current_row   = do_2D_norm ? id[dim_y] : 0;
            const int current_slice = id[dim];
            const int first_row     = do_2D_norm ? std::max(current_row - radius, min_top) : 0;
            const int last_row      = do_2D_norm ? std::min(current_row + radius, max_bottom) : 0;
            const int first_slice   = std::max(current_slice - radius, min_left);
            const int last_slice    = std::min(current_slice + radius, max_right);

            // Accumulate 2D In-Map values
            qint8x16_t accu = vdupq_n_qs8(0);
            for(int j = first_row; j <= last_row; ++j)
            {
                // Compute row displacement
                const int            row               = (j - current_row) * _input_squared->info()->strides_in_bytes()[dim_y];
                const uint8_t *const input_squared_ptr = input_squared.ptr() + row - (current_slice * input_squared_stride);
                for(int i = first_slice; i <= last_slice; ++i)
                {
                    accu = vqaddq_qs8(accu, vld1q_qs8(reinterpret_cast<const qint8_t *>(input_squared_ptr + i * input_squared_stride)));
                }
            }

            // Normalize
            const qint8x16_t accu_scale       = vqmlaq_qs8(kappa_vec, coeff_vec, accu, fixed_point_position);
            const qint8x16_t normalized       = vqpowq_qs8(accu_scale, beta_vec, fixed_point_position);
            const qint8x16_t normalized_pixel = vdivq_qs8(vld1q_qs8(reinterpret_cast<const qint8_t *>(input.ptr())), normalized, fixed_point_position);
            vst1q_qs8(reinterpret_cast<qint8_t *>(output.ptr()), normalized_pixel);
        },
        input, input_squared, output);
    }
    else if(dt == DataType::QS16)
    {
        const qint16x8_t coeff_vec = vdupq_n_qs16_f32(_norm_info.scale_coeff(), fixed_point_position);
        const qint16x8_t beta_vec  = vdupq_n_qs16_f32(_norm_info.beta(), fixed_point_position);
        const qint16x8_t kappa_vec = vdupq_n_qs16_f32(_norm_info.kappa(), fixed_point_position);

        execute_window_loop(window, [&](const Coordinates & id)
        {
            // Get range to normalize
            const int current_row   = do_2D_norm ? id[dim_y] : 0;
            const int current_slice = id[dim];
            const int first_row     = do_2D_norm ? std::max(current_row - radius, min_top) : 0;
            const int last_row      = do_2D_norm ? std::min(current_row + radius, max_bottom) : 0;
            const int first_slice   = std::max(current_slice - radius, min_left);
            const int last_slice    = std::min(current_slice + radius, max_right);

            // Accumulate 2D In-Map values
            qint16x8_t accu = vdupq_n_qs16(0);
            for(int j = first_row; j <= last_row; ++j)
            {
                // Compute row displacement
                const int            row               = (j - current_row) * _input_squared->info()->strides_in_bytes()[dim_y];
                const uint8_t *const input_squared_ptr = input_squared.ptr() + row - (current_slice * input_squared_stride);
                for(int i = first_slice; i <= last_slice; ++i)
                {
                    accu = vqaddq_qs16(accu, vld1q_qs16(reinterpret_cast<const qint16_t *>(input_squared_ptr + i * input_squared_stride)));
                }
            }

            // Normalize
            const qint16x8_t accu_scale       = vqmlaq_qs16(kappa_vec, coeff_vec, accu, fixed_point_position);
            const qint16x8_t normalized       = vqpowq_qs16(accu_scale, beta_vec, fixed_point_position);
            const qint16x8_t normalized_pixel = vdivq_qs16(vld1q_qs16(reinterpret_cast<const qint16_t *>(input.ptr())), normalized, fixed_point_position);
            vst1q_qs16(reinterpret_cast<qint16_t *>(output.ptr()), normalized_pixel);
        },
        input, input_squared, output);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not supported");
    }
}

void NENormalizationLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    // Run function
    (this->*_func)(window);
}
