/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#include "src/core/NEON/kernels/NEInstanceNormalizationLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/instancenorm/list.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
struct InstanceNormSelectorData
{
    DataType dt;
};

using InstanceNormSelctorPtr = std::add_pointer<bool(const InstanceNormSelectorData &data)>::type;
using InstanceNormUKernelPtr = std::add_pointer<void(ITensor      *input,
                                                     ITensor      *output,
                                                     float         gamma,
                                                     float         beta,
                                                     float         epsilon,
                                                     bool          use_mixed_precision,
                                                     const Window &window)>::type;

struct InstanceNormKernel
{
    const char                  *name;
    const InstanceNormSelctorPtr is_selected;
    InstanceNormUKernelPtr       ukernel;
};

static const InstanceNormKernel available_kernels[] = {
    {"fp32_neon_instancenorm", [](const InstanceNormSelectorData &data) { return data.dt == DataType::F32; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_instancenorm)},
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    {"fp16_neon_instancenorm", [](const InstanceNormSelectorData &data) { return data.dt == DataType::F16; },
     REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_instancenorm)},
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const InstanceNormKernel *get_implementation(const InstanceNormSelectorData &data)
{
    for (const auto &uk : available_kernels)
    {
        if (uk.is_selected(data))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, float gamma, float beta, float epsilon)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_UNUSED(gamma);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(epsilon == 0.f, "Epsilon must be different than 0");

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_layout() == DataLayout::NHWC,
                                    "NHWC data layout is not supported by the kernel directly");

    if (output != nullptr && output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_channels() != output->num_channels(),
                                        "Input and output have different number of channels");
    }
    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // We handle the planes manually
    Window win = calculate_max_window(*input, Steps(1));

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type());

    // NEInstanceNormalizationLayerKernel doesn't need padding so update_window_and_padding() can be skipped
    return std::make_pair(Status{}, win);
}
} // namespace

NEInstanceNormalizationLayerKernel::NEInstanceNormalizationLayerKernel()
    : _input(nullptr), _output(nullptr), _gamma(1), _beta(0), _epsilon(1e-12)
{
}

void NEInstanceNormalizationLayerKernel::configure(ITensor                                    *input,
                                                   ITensor                                    *output,
                                                   const InstanceNormalizationLayerKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _input               = input;
    _output              = output == nullptr ? input : output;
    _gamma               = info.gamma;
    _beta                = info.beta;
    _epsilon             = info.epsilon;
    _use_mixed_precision = info.use_mixed_precision;

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(_input->info(), _output->info(), _gamma, _beta, _epsilon));

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEInstanceNormalizationLayerKernel::validate(const ITensorInfo                          *input,
                                                    const ITensorInfo                          *output,
                                                    const InstanceNormalizationLayerKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, info.gamma, info.beta, info.epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(
        input->clone().get(), (output == nullptr ? input->clone().get() : output->clone().get()))));
    return Status{};
}

void NEInstanceNormalizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const auto *uk = get_implementation(InstanceNormSelectorData{_input->info()->data_type()});
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    uk->ukernel(_input, _output, _gamma, _beta, _epsilon, _use_mixed_precision, window);
}
} // namespace arm_compute
