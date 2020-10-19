/*
 * Copyright (c) 2020 Arm Limited.
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
#include "src/runtime/CL/gemm/CLGEMMKernelSelectionValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "src/core/CL/gemm/CLGEMMHelpers.h"

#include <map>
#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
CLGEMMKernelSelectionValhall::CLGEMMKernelSelectionValhall(GPUTarget gpu)
    : ICLGEMMKernelSelection(gpu)
{
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::select_kernel(const CLGEMMKernelSelectionParams &params)
{
    // _target could be used in the future to have a dedicated heuristic for each GPU IP
    ARM_COMPUTE_UNUSED(_target);

    using FunctionExecutorPtr = CLGEMMKernelType (CLGEMMKernelSelectionValhall::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);

    // Configurations for Valhall architectures
    static std::map<DataType, FunctionExecutorPtr> gemm_configs =
    {
        { DataType::F32, &CLGEMMKernelSelectionValhall::default_f32 },
        { DataType::F16, &CLGEMMKernelSelectionValhall::default_f16 },
        { DataType::QASYMM8, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QSYMM8, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMKernelSelectionValhall::default_q8 }
    };

    const DataType data_type = params.data_type;

    if(gemm_configs.find(data_type) != gemm_configs.end())
    {
        return (this->*gemm_configs[data_type])(params.m, params.n, params.k, params.b, params.is_rhs_constant);
    }

    ARM_COMPUTE_ERROR("Not supported data type");
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::default_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    return is_rhs_constant ? CLGEMMKernelType::RESHAPED_ONLY_RHS : CLGEMMKernelType::NATIVE_V1;
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::default_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    return is_rhs_constant ? CLGEMMKernelType::RESHAPED_ONLY_RHS : CLGEMMKernelType::NATIVE_V1;
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::default_q8(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    if(is_rhs_constant)
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
    }
    else
    {
        return CLGEMMKernelType::NATIVE;
    }
}
} // namespace cl_gemm
} // namespace arm_compute
