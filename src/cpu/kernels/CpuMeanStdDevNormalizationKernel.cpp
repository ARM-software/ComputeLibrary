/*
 * Copyright (c) 2019-2022, 2024-2025 Arm Limited.
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
#include "src/cpu/kernels/CpuMeanStdDevNormalizationKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

#include "src/common/utils/profile/acl_profile.h"
#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/meanstddevnorm/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{

static const std::vector<CpuMeanStdDevNormalizationKernel::MeanStdDevNormKernel> available_kernels = {
    {"fp32_neon_meanstddevnorm", [](const DataTypeSelectorData &data) { return data.dt == DataType::F32; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_meanstddevnorm)},
#ifdef ARM_COMPUTE_ENABLE_FP16
    {"fp16_neon_meanstddevnorm", [](const DataTypeSelectorData &data) { return data.dt == DataType::F16; },
     REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_meanstddevnorm)},
#endif // ARM_COMPUTE_ENABLE_FP16
    {"qasymm8_neon_meanstddevnorm", [](const DataTypeSelectorData &data) { return data.dt == DataType::QASYMM8; },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_qasymm8_meanstddevnorm)},
};

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() > 2, "Input tensor cannot have more than 2 dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_SIZE_UNSUPPORTED(input);

    // Checks performed when output is configured
    if ((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_SIZE_UNSUPPORTED(output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    if (output != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input);
    }

    // This kernel doesn't need padding. A left-over for loop on dimension X, we cannot have any read or write out of memory
    // For this reason num_elems_processed_per_iteration is set to 1
    Window win = calculate_max_window(*input, Steps());

    return std::make_pair(Status{}, win);
}
} // namespace

const std::vector<CpuMeanStdDevNormalizationKernel::MeanStdDevNormKernel> &
CpuMeanStdDevNormalizationKernel::get_available_kernels()
{
    return available_kernels;
}

void CpuMeanStdDevNormalizationKernel::configure(ITensorInfo *input, ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "CpuMeanStdDevNormalizationKernel::configure");
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    ARM_COMPUTE_ERROR_THROW_ON(
        CpuMeanStdDevNormalizationKernel::validate(input, (output != nullptr) ? output : nullptr, epsilon));

    _epsilon = epsilon;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input, (output == nullptr) ? nullptr : output);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);
}

Status CpuMeanStdDevNormalizationKernel::validate(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(
        validate_and_configure_window(input->clone().get(), (output != nullptr) ? output->clone().get() : nullptr)
            .first);
    return Status{};
}

void CpuMeanStdDevNormalizationKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU,
                            "CpuMeanStdDevNormalizationKernel::run_op");
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ITensor    *input  = tensors.get_tensor(TensorType::ACL_SRC);
    ITensor    *output = tensors.get_tensor(TensorType::ACL_DST);
    const auto *uk     = get_implementation<DataTypeSelectorData>(DataTypeSelectorData{output->info()->data_type()});
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);
    uk->ukernel(input, output, _epsilon, window);
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
