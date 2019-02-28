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
#include "arm_compute/core/NEON/kernels/NENormalizationLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

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

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *input_squared, ITensorInfo *output, const NormalizationLayerInfo &norm_info)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, *input->clone());

    const unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    const unsigned int norm_idx              = get_normalization_dimension_index(input->data_layout(), norm_info);
    const bool         is_norm_accross_width = norm_idx == 0;

    const unsigned int border_width = is_norm_accross_width ? num_elems_processed_per_iteration - 1 : 0;
    const BorderSize   border_size  = BorderSize(0, border_width);

    // Configure window
    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool   window_changed = false;

    if(is_norm_accross_width)
    {
        AccessWindowStatic input_access(input, -border_size.left, 0, input->dimension(0) + border_size.right, 0);
        AccessWindowStatic input_squared_access(input_squared, -border_size.left, 0, input->dimension(0) + border_size.right, 0);
        window_changed = window_changed || update_window_and_padding(win, input_access, input_squared_access);
    }
    else
    {
        AccessWindowHorizontal input_access(input, -border_size.left, num_elems_processed_per_iteration);
        AccessWindowHorizontal input_squared_access(input_squared, -border_size.left, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, input_access, input_squared_access);
    }

    if(output->total_size() != 0)
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, output_access);
        output_access.set_valid_region(win, input->valid_region());
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

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
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, input_squared, output);
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), *input->info());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), input_squared->info(), output->info(), norm_info));

    const unsigned int num_elems_processed_per_iteration = 16 / input->info()->element_size();

    const unsigned int norm_idx              = get_normalization_dimension_index(input->info()->data_layout(), norm_info);
    const bool         is_norm_accross_width = norm_idx == 0;
    const unsigned int border_width          = is_norm_accross_width ? num_elems_processed_per_iteration - 1 : 0;

    _input         = input;
    _input_squared = input_squared;
    _output        = output;
    _norm_info     = norm_info;
    _border_size   = BorderSize(0, border_width);

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
    auto win_config = validate_and_configure_window(input->info(), input_squared->info(), output->info(), norm_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

template <typename T, unsigned int S, unsigned int dim, bool do_2D_norm>
void NENormalizationLayerKernel::normalize_float(const Window &window)
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    Iterator input(_input, window);
    Iterator input_squared(_input_squared, window);
    Iterator output(_output, window);

    const int dim_y                = _input->info()->data_layout() == DataLayout::NCHW ? 1 : 2;
    const int radius               = _norm_info.norm_size() / 2;
    const int input_squared_stride = _input_squared->info()->strides_in_bytes()[dim];
    // We account padding across X only and we iterate over rows
    const int min_left   = (dim == 2) ? 0 : -static_cast<int>(border_size().left);
    const int max_right  = _input->info()->dimension(dim) - 1;
    const int max_bottom = _input->info()->dimension(dim_y) - 1;

    const auto coeff_vec = wrapper::vdup_n(static_cast<T>(_norm_info.scale_coeff()), ExactTagType{});
    const auto beta_vec  = wrapper::vdup_n(static_cast<T>(_norm_info.beta()), ExactTagType{});
    const auto kappa_vec = wrapper::vdup_n(static_cast<T>(_norm_info.kappa()), ExactTagType{});

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Get range to normalize
        const int current_row   = do_2D_norm ? id[dim_y] : 0;
        const int current_slice = id[dim];
        const int first_row     = do_2D_norm ? std::max(current_row - radius, 0) : 0;
        const int last_row      = do_2D_norm ? std::min(current_row + radius, max_bottom) : 0;
        const int first_slice   = std::max(current_slice - radius, min_left);
        const int last_slice    = std::min(current_slice + radius, max_right);

        // Accumulate 2D In-Map values
        auto accu = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
        for(int j = first_row; j <= last_row; j++)
        {
            // Compute row displacement
            const int            row               = (j - current_row) * _input_squared->info()->strides_in_bytes()[dim_y];
            const uint8_t *const input_squared_ptr = input_squared.ptr() + row - (current_slice * input_squared_stride);
            for(int i = first_slice; i <= last_slice; ++i)
            {
                accu = wrapper::vadd(accu, wrapper::vloadq(reinterpret_cast<const T *>(input_squared_ptr + i * input_squared_stride)));
            }
        }

        // Normalize
        const auto normalized       = wrapper::vpow(wrapper::vmla(kappa_vec, coeff_vec, accu), beta_vec);
        const auto normalized_pixel = wrapper::vmul(wrapper::vloadq(reinterpret_cast<const T *>(input.ptr())), wrapper::vinv(normalized));
        wrapper::vstore(reinterpret_cast<T *>(output.ptr()), normalized_pixel);
    },
    input, input_squared, output);
}

Status NENormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *input_squared, const ITensorInfo *output, const NormalizationLayerInfo norm_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, input_squared, output, norm_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), input_squared->clone().get(), output->clone().get(), norm_info).first);

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
