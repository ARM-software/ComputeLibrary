/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NENormalizationLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/NormalizationHelpers.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *input_squared, const ITensorInfo *output, const NormalizationLayerInfo &norm_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, input_squared, output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, input_squared);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, input_squared);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!(norm_info.norm_size() % 2), "Normalization size should be odd");

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    }

    return Status{};
}

} // namespace

NENormalizationLayerKernel::NENormalizationLayerKernel()
    : _func(nullptr), _input(nullptr), _input_squared(nullptr), _output(nullptr), _norm_info(NormType::IN_MAP_1D)
{
}

void NENormalizationLayerKernel::configure(const ITensor *input, const ITensor *input_squared, ITensor *output, NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, input_squared, output);
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), *input->info());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), input_squared->info(), output->info(), norm_info));

    const unsigned int norm_idx = get_normalization_dimension_index(input->info()->data_layout(), norm_info);

    _input         = input;
    _input_squared = input_squared;
    _output        = output;
    _norm_info     = norm_info;

    switch(_input->info()->data_type())
    {
        case DataType::F32:
        {
            switch(norm_idx)
            {
                case 0:
                {
                    if(norm_info.type() == NormType::IN_MAP_2D)
                    {
                        _func = &NENormalizationLayerKernel::normalize_float<float, 4, 0, true>;
                    }
                    else
                    {
                        _func = &NENormalizationLayerKernel::normalize_float<float, 4, 0, false>;
                    }
                    break;
                }
                case 1:
                    if(norm_info.type() == NormType::IN_MAP_2D)
                    {
                        _func = &NENormalizationLayerKernel::normalize_float<float, 4, 1, true>;
                    }
                    else
                    {
                        _func = &NENormalizationLayerKernel::normalize_float<float, 4, 1, false>;
                    }
                    break;
                case 2:
                    _func = &NENormalizationLayerKernel::normalize_float<float, 4, 2, false>;
                    break;
                default:
                    break;
            }
            break;
        }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
        {
            switch(norm_idx)
            {
                case 0:
                {
                    if(norm_info.type() == NormType::IN_MAP_2D)
                    {
                        _func = &NENormalizationLayerKernel::normalize_float<float16_t, 8, 0, true>;
                    }
                    else
                    {
                        _func = &NENormalizationLayerKernel::normalize_float<float16_t, 8, 0, false>;
                    }
                    break;
                }
                case 1:
                    if(norm_info.type() == NormType::IN_MAP_2D)
                    {
                        _func = &NENormalizationLayerKernel::normalize_float<float16_t, 8, 1, true>;
                    }
                    else
                    {
                        _func = &NENormalizationLayerKernel::normalize_float<float16_t, 8, 1, false>;
                    }
                    break;
                case 2:
                    _func = &NENormalizationLayerKernel::normalize_float<float16_t, 8, 2, false>;
                    break;
                default:
                    break;
            }
            break;
        }
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

    // Configure kernel window
    Window      win = calculate_max_window(*input->info(), Steps());
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));
    INEKernel::configure(win);
}

template <typename T, unsigned int S, unsigned int dim, bool do_2D_norm>
void NENormalizationLayerKernel::normalize_float(const Window &window)
{
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = S;

    Iterator input(_input, win);
    Iterator input_squared(_input_squared, win);
    Iterator output(_output, win);

    const int dim_y                      = _input->info()->data_layout() == DataLayout::NCHW ? 1 : 2;
    const int radius                     = _norm_info.norm_size() / 2;
    const int input_squared_stride_x     = _input_squared->info()->strides_in_bytes()[0];
    const int input_squared_stride_slice = _input_squared->info()->strides_in_bytes()[dim];
    const int input_squared_stride_row   = _input_squared->info()->strides_in_bytes()[dim_y];

    const int max_right  = _input->info()->dimension(dim) - 1;
    const int max_bottom = _input->info()->dimension(dim_y) - 1;

    const auto coeff_vec = wrapper::vdup_n(static_cast<T>(_norm_info.scale_coeff()), ExactTagType{});
    const auto beta_vec  = wrapper::vdup_n(static_cast<T>(_norm_info.beta()), ExactTagType{});
    const auto kappa_vec = wrapper::vdup_n(static_cast<T>(_norm_info.kappa()), ExactTagType{});

    auto sequential_normalization = [&](const int x, const Coordinates & id, const int current_row, const int first_row, const int last_row, const T * input_ptr, const uint8_t *input_squared_start_ptr,
                                        T * output_ptr)
    {
        const int current_slice = dim == 0 ? x : id[dim];
        const int first_slice   = std::max(current_slice - radius, 0);
        const int last_slice    = std::min(current_slice + radius, max_right);

        const uint8_t *const input_squared_x_ptr = input_squared_start_ptr + x * input_squared_stride_x;
        // Accumulate 2D In-Map values
        auto accu = static_cast<T>(0.f);
        for(int j = first_row; j <= last_row; ++j)
        {
            // Compute row displacement
            const uint8_t *const input_squared_ptr = input_squared_x_ptr + (j - current_row) * input_squared_stride_row;
            for(int i = first_slice; i <= last_slice; ++i)
            {
                accu += *reinterpret_cast<const T *>(input_squared_ptr + (i - current_slice) * input_squared_stride_slice);
            }
        }

        // Normalize
        const auto normalized       = std::pow(accu * static_cast<T>(_norm_info.scale_coeff()) + static_cast<T>(_norm_info.kappa()), _norm_info.beta());
        const auto normalized_pixel = (*(input_ptr + x)) / normalized;
        *(output_ptr + x)           = normalized_pixel;
    };

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        auto       output_ptr = reinterpret_cast<T *>(output.ptr());

        // Get range to normalize
        const int current_row = do_2D_norm ? id[dim_y] : 0;
        const int first_row   = do_2D_norm ? std::max(current_row - radius, 0) : 0;
        const int last_row    = do_2D_norm ? std::min(current_row + radius, max_bottom) : 0;

        int x = window_start_x;
        // Compute serially starting elements for the case x dimension is width
        for(; x < radius && x < window_end_x && dim == 0; ++x)
        {
            sequential_normalization(x, id, current_row, first_row, last_row, input_ptr, input_squared.ptr(), output_ptr);
        }

        // Compute vectorized
        for(; x <= window_end_x - window_step_x - radius; x += window_step_x)
        {
            const int current_slice = dim == 0 ? x : id[dim];
            const int first_slice   = std::max(current_slice - radius, 0);
            const int last_slice    = std::min(current_slice + radius, max_right);

            const uint8_t *const input_squared_x_ptr = input_squared.ptr() + x * input_squared_stride_x;
            // Accumulate 2D In-Map values
            auto accu = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
            for(int j = first_row; j <= last_row; ++j)
            {
                // Compute row displacement
                const uint8_t *const input_squared_ptr = input_squared_x_ptr + (j - current_row) * input_squared_stride_row;
                for(int i = first_slice; i <= last_slice; ++i)
                {
                    accu = wrapper::vadd(accu, wrapper::vloadq(reinterpret_cast<const T *>(input_squared_ptr + (i - current_slice) * input_squared_stride_slice)));
                }
            }

            // Normalize
            const auto normalized       = wrapper::vpow(wrapper::vmla(kappa_vec, coeff_vec, accu), beta_vec);
            const auto normalized_pixel = wrapper::vmul(wrapper::vloadq(input_ptr + x), wrapper::vinv(normalized));
            wrapper::vstore(reinterpret_cast<T *>(output_ptr + x), normalized_pixel);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            sequential_normalization(x, id, current_row, first_row, last_row, input_ptr, input_squared.ptr(), output_ptr);
        }
    },
    input, input_squared, output);
}

Status NENormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *input_squared, const ITensorInfo *output, const NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, input_squared, output, norm_info));

    return Status{};
}

void NENormalizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    // Run function
    (this->*_func)(window);
}
} // namespace arm_compute
