/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/NormalizationHelpers.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/norm_layer/generic/neon/impl.h"
#include "src/cpu/kernels/norm_layer/generic/neon/list.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo            *input,
                          const ITensorInfo            *input_squared,
                          const ITensorInfo            *output,
                          const NormalizationLayerInfo &norm_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, input_squared, output);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, input_squared);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, input_squared);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!(norm_info.norm_size() % 2), "Normalization size should be odd");

    // Checks performed when output is configured
    if (output->total_size() != 0)
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

void NENormalizationLayerKernel::configure(const ITensor         *input,
                                           const ITensor         *input_squared,
                                           ITensor               *output,
                                           NormalizationLayerInfo norm_info)
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
    switch (_input->info()->data_type())
    {
        case DataType::F32:
        {
            switch (norm_idx)
            {
                case 0:
                {
                    if (norm_info.type() == NormType::IN_MAP_2D)
                    {
                        _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_0_2D);
                    }
                    else
                    {
                        _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_0);
                    }
                    break;
                }
                case 1:
                    if (norm_info.type() == NormType::IN_MAP_2D)
                    {
                        _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_1_2D);
                    }
                    else
                    {
                        _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_1);
                    }
                    break;
                case 2:
                    _func = REGISTER_FP32_NEON(cpu::neon_normalize_float32_4_2);
                    break;
                default:
                    break;
            }
            break;
        }
#ifdef ARM_COMPUTE_ENABLE_FP16
        case DataType::F16:
        {
            switch (norm_idx)
            {
                case 0:
                {
                    if (norm_info.type() == NormType::IN_MAP_2D)
                    {
                        _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_0_2D);
                    }
                    else
                    {
                        _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_0);
                    }
                    break;
                }
                case 1:
                    if (norm_info.type() == NormType::IN_MAP_2D)
                    {
                        _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_1_2D);
                    }
                    else
                    {
                        _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_1);
                    }
                    break;
                case 2:
                    _func = REGISTER_FP16_NEON(cpu::neon_normalize_float16_8_2);
                    break;
                default:
                    break;
            }
            break;
        }
#endif /* ARM_COMPUTE_ENABLE_FP16 */
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    INEKernel::configure(win);
}

Status NENormalizationLayerKernel::validate(const ITensorInfo           *input,
                                            const ITensorInfo           *input_squared,
                                            const ITensorInfo           *output,
                                            const NormalizationLayerInfo norm_info)
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
    (*_func)(window, _input, _input_squared, _output, _norm_info);
}
} // namespace arm_compute
