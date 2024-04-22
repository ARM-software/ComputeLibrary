/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "src/core/NEON/kernels/NERangeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/range/list.h"

namespace arm_compute
{
namespace
{
struct RangeSelectorData
{
    DataType dt;
};

using RangeSelectorPtr = std::add_pointer<bool(const RangeSelectorData &data)>::type;
using RangeUKernelPtr  = std::add_pointer<void(ITensor *, float, float, const Window &)>::type;

struct RangeUKernel
{
    const char            *name;
    const RangeSelectorPtr is_selected;
    RangeUKernelPtr        ukernel;
};

static const RangeUKernel available_kernels[] = {
    {"fp16_neon_range", [](const RangeSelectorData &data) { return data.dt == DataType::F16; },
     REGISTER_FP16_NEON(arm_compute::cpu::fp16_neon_range_function)},
    {"f32_neon_range", [](const RangeSelectorData &data) { return data.dt == DataType::F32; },
     REGISTER_FP32_NEON(arm_compute::cpu::fp32_neon_range_function)},
    {"u8_neon_range", [](const RangeSelectorData &data) { return data.dt == DataType::U8; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::u8_neon_range_function)},
    {"u16_neon_range", [](const RangeSelectorData &data) { return data.dt == DataType::U16; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::u16_neon_range_function)},
    {"u32_neon_range", [](const RangeSelectorData &data) { return data.dt == DataType::U32; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::u32_neon_range_function)},
    {"s8_neon_range", [](const RangeSelectorData &data) { return data.dt == DataType::S8; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::s8_neon_range_function)},
    {"s16_neon_range", [](const RangeSelectorData &data) { return data.dt == DataType::S16; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::s16_neon_range_function)},
    {"s32_neon_range", [](const RangeSelectorData &data) { return data.dt == DataType::S32; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::s32_neon_range_function)},
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const RangeUKernel *get_implementation(const RangeSelectorData &data)
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

Status validate_arguments(const ITensorInfo &output, const float start, const float end, const float step)
{
    const auto *uk = get_implementation(RangeSelectorData{output.data_type()});
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((start == end), "start of the requested sequence must not be equal to the end");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((start < end) && (step <= 0)), "step must be greater than 0 when start < end");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(((start > end) && (step >= 0)), "step must be less than 0 when start > end");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(start, output.data_type(), output.quantization_info()),
                                    "start value is outside the range of the data type");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(end, output.data_type(), output.quantization_info()),
                                    "end value is outside the range of the data type");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!check_value_range(step, output.data_type(), output.quantization_info()),
                                    "step value is outside the range of the data type");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((start == end), "start of the requested sequence must not be equal to the end");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output.num_dimensions() != 1, "Output has to be a 1-D tensor");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(output.tensor_shape().total_size() < num_of_elements_in_range(start, end, step),
                                    "Output tensor size is incorrect");

    return Status{};
}
} // namespace

NERangeKernel::NERangeKernel() : _start(0), _end(1), _step(1), _output(nullptr)
{
}

void NERangeKernel::configure(ITensor *output, float start, float end, float step)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*(output->info()), start, end, step));

    // Auto initialize output if not initialized
    auto_init_if_empty(*output->info(), TensorShape(num_of_elements_in_range(start, end, step)), 1,
                       output->info()->data_type(), output->info()->quantization_info());

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());

    _start  = start;
    _end    = end;
    _step   = step;
    _output = output;

    INEKernel::configure(win);
}

Status NERangeKernel::validate(const ITensorInfo *output, float start, float end, float step)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*output, start, end, step));

    return Status{};
}

void NERangeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    const auto *uk = get_implementation(RangeSelectorData{_output->info()->data_type()});

    uk->ukernel(_output, _start, _step, window);
}
} // namespace arm_compute
