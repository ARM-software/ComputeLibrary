/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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
#include "src/cpu/kernels/CpuDequantizeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NESymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/dequantize/generic/neon/list.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::QSYMM8_PER_CHANNEL, DataType::QSYMM8,
                                                         DataType::QSYMM16);

    if (dst->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(dst);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }

    return Status{};
}
} // namespace

void CpuDequantizeKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, src->tensor_shape(), 1, DataType::F32);

    ICpuKernel::configure(win);

    switch (dst->data_type())
    {
        case DataType::F32:
            _func = REGISTER_FP32_NEON(fp32_run_dequantization_core);
            break;
#ifdef ARM_COMPUTE_ENABLE_FP16
        case DataType::F16:
            _func = REGISTER_FP16_NEON(fp16_run_dequantization_core);
            break;
#endif /* ARM_COMPUTE_ENABLE_FP16 */
        default:
            ARM_COMPUTE_ERROR("Unsupported data type.");
    }
}

Status CpuDequantizeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
    return Status{};
}

void CpuDequantizeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);
    _func(src, dst, window);
}
const char *CpuDequantizeKernel::name() const
{
    return "CpuDequantizeKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
