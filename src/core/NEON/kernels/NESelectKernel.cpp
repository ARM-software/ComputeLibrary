/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#include "src/core/NEON/kernels/NESelectKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/select/list.h"

#include <arm_neon.h>
#include <map>
#include <string>

namespace arm_compute
{
namespace
{

struct SelectKernelSelectorData
{
    DataType dt;
    bool     is_same_rank;
};

using SelectorPtr = std::add_pointer<bool(const SelectKernelSelectorData &data)>::type;
using KernelPtr =
    std::add_pointer<void(const ITensor *, const ITensor *, const ITensor *, ITensor *, const Window &)>::type;

struct SelectKernelSelector
{
    const char       *name;
    const SelectorPtr is_selected;
    KernelPtr         ukernel;
};

static const SelectKernelSelector available_kernels[] = {
    {"neon_s8_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::S8 && data.is_same_rank == true; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_s8_select_same_rank)},
    {"neon_s16_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::S16 && data.is_same_rank == true; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_s16_select_same_rank)},
    {"neon_s32_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::S32 && data.is_same_rank == true; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_s32_select_same_rank)},
    {"neon_u8_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::U8 && data.is_same_rank == true; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_u8_select_same_rank)},
    {"neon_u16_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::U16 && data.is_same_rank == true; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_u16_select_same_rank)},
    {"neon_u32_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::U32 && data.is_same_rank == true; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_u32_select_same_rank)},
    {"neon_s8_not_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::S8 && data.is_same_rank == false; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_s8_select_not_same_rank)},
    {"neon_s16_not_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::S16 && data.is_same_rank == false; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_s16_select_not_same_rank)},
    {"neon_s32_not_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::S32 && data.is_same_rank == false; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_s32_select_not_same_rank)},
    {"neon_u8_not_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::U8 && data.is_same_rank == false; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_u8_select_not_same_rank)},
    {"neon_u16_not_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::U16 && data.is_same_rank == false; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_u16_select_not_same_rank)},
    {"neon_u32_not_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::U32 && data.is_same_rank == false; },
     REGISTER_INTEGER_NEON(arm_compute::cpu::neon_u32_select_not_same_rank)},
    {"neon_f16_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::F16 && data.is_same_rank == true; },
     REGISTER_FP16_NEON(arm_compute::cpu::neon_f16_select_same_rank)},
    {"neon_f16_not_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::F16 && data.is_same_rank == false; },
     REGISTER_FP16_NEON(arm_compute::cpu::neon_f16_select_not_same_rank)},
    {"neon_f32_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::F32 && data.is_same_rank == true; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_f32_select_same_rank)},
    {"neon_f32_not_same_rank",
     [](const SelectKernelSelectorData &data) { return data.dt == DataType::F32 && data.is_same_rank == false; },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_f32_select_not_same_rank)},
};

const SelectKernelSelector *get_implementation(const SelectKernelSelectorData &data)
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

} // namespace

NESelectKernel::NESelectKernel()
    : /*_function(nullptr), */ _c(nullptr), _x(nullptr), _y(nullptr), _output(nullptr), _has_same_rank(false)
{
}

void NESelectKernel::configure(const ITensor *c, const ITensor *x, const ITensor *y, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(c, x, y, output);

    // Auto initialize output if not initialized
    auto_init_if_empty(*output->info(), x->info()->tensor_shape(), 1, x->info()->data_type());
    ARM_COMPUTE_ERROR_THROW_ON(validate(c->info(), x->info(), y->info(), output->info()));

    _c             = c;
    _x             = x;
    _y             = y;
    _output        = output;
    _has_same_rank = (c->info()->tensor_shape().num_dimensions() == x->info()->tensor_shape().num_dimensions());

    Window win = calculate_max_window(*x->info());
    INEKernel::configure(win);
}

Status
NESelectKernel::validate(const ITensorInfo *c, const ITensorInfo *x, const ITensorInfo *y, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(c, x, y);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(x);
    ARM_COMPUTE_RETURN_ERROR_ON(x->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(x, y);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(x, y);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(c, 1, DataType::U8);

    const bool is_same_rank = (c->tensor_shape().num_dimensions() == x->tensor_shape().num_dimensions());
    ARM_COMPUTE_RETURN_ERROR_ON(is_same_rank && (x->tensor_shape() != c->tensor_shape()));
    ARM_COMPUTE_RETURN_ERROR_ON(!is_same_rank &&
                                ((c->tensor_shape().num_dimensions() > 1) ||
                                 (c->tensor_shape().x() != x->tensor_shape()[x->tensor_shape().num_dimensions() - 1])));

    if (output != nullptr && output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(x, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(x, output);
    }

    return Status{};
}

void NESelectKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_output == nullptr);
    ARM_COMPUTE_ERROR_ON(_output->info() == nullptr);

    const auto *uk = get_implementation(SelectKernelSelectorData{_output->info()->data_type(), _has_same_rank});
    ARM_COMPUTE_ERROR_ON(uk == nullptr);
    ARM_COMPUTE_ERROR_ON(uk->ukernel == nullptr);
    uk->ukernel(_c, _x, _y, _output, window);
}
} // namespace arm_compute
