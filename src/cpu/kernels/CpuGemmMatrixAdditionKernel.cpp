/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#include "src/cpu/kernels/CpuGemmMatrixAdditionKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/cpu/kernels/gemm_matrix_add/list.h"
namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuGemmMatrixAdditionKernel::GemmMatrixAddKernel> available_kernels = {
    {"neon_fp32_gemm_matrix_add", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F32); },
     REGISTER_FP32_NEON(neon_fp32_gemm_matrix_add)},
    {"neon_fp16_gemm_matrix_add",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F16) && data.isa.fp16; },
     REGISTER_FP16_NEON(neon_fp16_gemm_matrix_add)},

};
} // namespace

void CpuGemmMatrixAdditionKernel::configure(const ITensorInfo *src, ITensorInfo *dst, float beta)
{
    ARM_COMPUTE_UNUSED(dst);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemmMatrixAdditionKernel::validate(src, dst, beta));

    _beta         = beta;
    const auto uk = CpuGemmMatrixAdditionKernel::get_implementation(
        DataTypeISASelectorData{src->data_type(), CPUInfo::get().get_isa()});
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);
    _func = uk->ukernel;
    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());
    ICPPKernel::configure(win);
}

Status CpuGemmMatrixAdditionKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, float beta)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_UNUSED(beta);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);

    if (dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
    }
    return Status{};
}

void CpuGemmMatrixAdditionKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST);

    if (_beta != 0.0f)
    {
        (*_func)(src, dst, window, _beta);
    }
}

const char *CpuGemmMatrixAdditionKernel::name() const
{
    return "CpuGemmMatrixAdditionKernel";
}

const std::vector<CpuGemmMatrixAdditionKernel::GemmMatrixAddKernel> &
CpuGemmMatrixAdditionKernel::get_available_kernels()
{
    return available_kernels;
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
