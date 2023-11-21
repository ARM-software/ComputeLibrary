/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#include "src/core/NEON/kernels/NEL2NormalizeLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEMath.h"
#include "src/cpu/kernels/l2normlayer/list.h"

#include <arm_neon.h>
#include <cmath>

namespace arm_compute
{
namespace
{
constexpr int max_input_tensor_dim = 3;

struct L2NormalizeLayerSelectorData
{
    DataType            dt;
    unsigned int        actual_axis;
    cpuinfo::CpuIsaInfo isa;
};

using L2NormalizeLayerKernelSelctorPtr = std::add_pointer<bool(const L2NormalizeLayerSelectorData &data)>::type;

using L2NormalizeLayerPtr = std::add_pointer<void(
    const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window, size_t axis)>::type;

struct L2NormalizeLayerKernel
{
    const char                            *name;
    const L2NormalizeLayerKernelSelctorPtr is_selected;
    L2NormalizeLayerPtr                    ukernel;
};

static const L2NormalizeLayerKernel available_kernels[] = {
    {"fp32_neon_l2normalize_x",
     [](const L2NormalizeLayerSelectorData &data)
     { return data.dt == DataType::F32 && data.actual_axis == Window::DimX; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_l2_normalize_x)},
    {"fp32_neon_l2normalize_yz",
     [](const L2NormalizeLayerSelectorData &data)
     { return data.dt == DataType::F32 && data.actual_axis != Window::DimX; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_l2_normalize_yz)},
    {
        "fp16_neon_l2normalize_x",
        [](const L2NormalizeLayerSelectorData &data)
        { return data.dt == DataType::F16 && data.isa.fp16 && data.actual_axis == Window::DimX; },
        REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_l2_normalize_x),
    },
    {
        "fp16_neon_l2normalize_yz",
        [](const L2NormalizeLayerSelectorData &data)
        { return data.dt == DataType::F16 && data.isa.fp16 && data.actual_axis != Window::DimX; },
        REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_l2_normalize_yz),
    },
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const L2NormalizeLayerKernel *get_implementation(const L2NormalizeLayerSelectorData &data)
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

Status
validate_arguments(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, int axis, float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);

    const uint32_t actual_axis = wrap_around(axis, max_input_tensor_dim);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, sum);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(actual_axis > 2, "Actual axis greater than 2 is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(actual_axis >= TensorShape::num_max_dimensions,
                                    "Actual normalization axis greater than max number of dimensions");

    // Reduce shape on axis
    TensorShape sum_shape = input->tensor_shape();
    sum_shape.set(actual_axis, 1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(sum->tensor_shape(), sum_shape);

    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(input->tensor_shape(), output->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    Window win = calculate_max_window(*input, Steps());

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type());

    // NEL2NormalizeLayerKernel doesn't need padding so update_window_and_padding() can be skipped

    return std::make_tuple(Status{}, win);
}
} // namespace

NEL2NormalizeLayerKernel::NEL2NormalizeLayerKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr), _actual_axis(0), _epsilon(1e-12)
{
}

void NEL2NormalizeLayerKernel::configure(
    const ITensor *input, const ITensor *sum, ITensor *output, int axis, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), sum->info(), output->info(), axis, epsilon));

    _input       = input;
    _sum         = sum;
    _output      = output;
    _actual_axis = wrap_around(axis, max_input_tensor_dim);
    _epsilon     = epsilon;

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEL2NormalizeLayerKernel::validate(
    const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, int axis, float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, sum, output, axis, epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(
        std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));

    return Status{};
}

void NEL2NormalizeLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    if (_actual_axis > 2)
    {
        ARM_COMPUTE_ERROR("Unsupported normalization axis");
    }

    const auto *uk = get_implementation(
        L2NormalizeLayerSelectorData{_output->info()->data_type(), _actual_axis, CPUInfo::get().get_isa()});
    ARM_COMPUTE_ERROR_ON(uk == nullptr);
    ARM_COMPUTE_ERROR_ON(uk->ukernel == nullptr);

    uk->ukernel(_input, _sum, _output, _epsilon, window, _actual_axis);
}
} // namespace arm_compute
