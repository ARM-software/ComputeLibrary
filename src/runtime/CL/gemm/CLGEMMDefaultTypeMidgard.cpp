/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "src/runtime/CL/gemm/CLGEMMDefaultTypeMidgard.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"

#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"

#include <map>
#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
CLGEMMDefaultTypeMidgard::CLGEMMDefaultTypeMidgard(GPUTarget gpu) : ICLGEMMKernelSelection(gpu)
{
}

CLGEMMKernelType CLGEMMDefaultTypeMidgard::select_kernel(const CLGEMMKernelSelectionParams &params)
{
    // _target could be used in the future to have a dedicated heuristic for each GPU IP
    ARM_COMPUTE_UNUSED(_target);

    using FunctionExecutorPtr = CLGEMMKernelType (CLGEMMDefaultTypeMidgard::*)(
        unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);

    // Configurations for Midgard architectures
    static std::map<DataType, FunctionExecutorPtr> gemm_configs = {
        {DataType::F32, &CLGEMMDefaultTypeMidgard::default_f32},
        {DataType::F16, &CLGEMMDefaultTypeMidgard::default_f16},
        {DataType::QASYMM8, &CLGEMMDefaultTypeMidgard::default_q8},
        {DataType::QASYMM8_SIGNED, &CLGEMMDefaultTypeMidgard::default_q8},
        {DataType::QSYMM8, &CLGEMMDefaultTypeMidgard::default_q8},
        {DataType::QSYMM8_PER_CHANNEL, &CLGEMMDefaultTypeMidgard::default_q8}};

    const DataType data_type = params.data_type;

    if (gemm_configs.find(data_type) != gemm_configs.end())
    {
        return (this->*gemm_configs[data_type])(params.m, params.n, params.k, params.b, params.is_rhs_constant);
    }

    ARM_COMPUTE_ERROR("Not supported data type");
}

CLGEMMKernelType CLGEMMDefaultTypeMidgard::default_f32(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(n, k, b);

    // We reshape the matrices only if we do not have the vector-by-matrix case and we reshape the matrix B only once
    return ((m != 1) && is_rhs_constant) ? CLGEMMKernelType::RESHAPED : CLGEMMKernelType::NATIVE;
}

CLGEMMKernelType CLGEMMDefaultTypeMidgard::default_f16(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(n, k, b);

    // We reshape the matrices only if we do not have the vector-by-matrix case and we reshape the matrix B only once
    return ((m != 1) && is_rhs_constant) ? CLGEMMKernelType::RESHAPED : CLGEMMKernelType::NATIVE;
}

CLGEMMKernelType CLGEMMDefaultTypeMidgard::default_q8(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b, is_rhs_constant);

    return CLGEMMKernelType::NATIVE;
}
} // namespace cl_gemm
} // namespace arm_compute
