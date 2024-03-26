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
#include "src/gpu/cl/kernels/gemm/native/ClGemmDefaultConfigNativeMidgard.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"

#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"

#include <utility>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace gemm
{
ClGemmDefaultConfigNativeMidgard::ClGemmDefaultConfigNativeMidgard(GPUTarget gpu) : IClGemmKernelConfig(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigNativeMidgard::configure(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (
        ClGemmDefaultConfigNativeMidgard::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_default(nullptr, nullptr,
                                                                        &ClGemmDefaultConfigNativeMidgard::default_q8);

    auto func = configs_default.get_function(data_type);
    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not support for GEMM");
    return (this->*func)(m, n, k, b);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigNativeMidgard::default_q8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    const unsigned int m0 = std::min(m, static_cast<unsigned int>(4));
    const unsigned int n0 = std::min(n, static_cast<unsigned int>(4));

    return configure_lhs_rhs_info(m, n, m0, n0, 2, 1, 1, false, false, false, false);
}
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
