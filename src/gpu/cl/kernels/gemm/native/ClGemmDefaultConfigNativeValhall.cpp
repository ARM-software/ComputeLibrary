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
#include "src/gpu/cl/kernels/gemm/native/ClGemmDefaultConfigNativeValhall.h"

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
ClGemmDefaultConfigNativeValhall::ClGemmDefaultConfigNativeValhall(GPUTarget gpu)
    : IClGemmKernelConfig(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigNativeValhall::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (ClGemmDefaultConfigNativeValhall::*)(unsigned int m, unsigned int n, unsigned int k,
                                             unsigned int b);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_default(&ClGemmDefaultConfigNativeValhall::configure_G77_f32,
                                                                        &ClGemmDefaultConfigNativeValhall::configure_G77_f16,
                                                                        &ClGemmDefaultConfigNativeValhall::configure_G77_u8);

    auto func = configs_default.get_function(data_type);
    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not support for GEMM");
    return (this->*func)(m, n, k, b);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigNativeValhall::configure_G77_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        if(n < 2048)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 1, false, false, false, false);
        }
        else if(n >= 2048 && n < 8192)
        {
            return configure_lhs_rhs_info(m, n, 1, 4, 4, 1, 1, false, false, false, false);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 1, 8, 4, 1, 1, false, false, false, false);
        }
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 5, 4, 2, 1, 1, false, false, false, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigNativeValhall::configure_G77_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        if(n < 2048)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 1, false, false, false, false);
        }
        else if(n >= 2048 && n < 8192)
        {
            return configure_lhs_rhs_info(m, n, 1, 4, 4, 1, 1, false, false, false, false);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 1, 8, 4, 1, 1, false, false, false, false);
        }
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 8, 2, 1, 1, false, false, false, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigNativeValhall::configure_G77_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(dot8_supported(CLKernelLibrary::get().get_device()))
    {
        if(m == 1)
        {
            if(n < 2048)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 1, false, false, false, false);
            }
            else if(n >= 2048 && n < 16384)
            {
                return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, 1, false, false, false, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 8, 16, 1, 1, false, false, false, false);
            }
        }
        else
        {
            if(m < 64)
            {
                return configure_lhs_rhs_info(m, n, 2, 2, 16, 1, 1, false, false, false, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 5, 2, 16, 1, 1, false, false, false, false);
            }
        }
    }
    else
    {
        if(m == 1)
        {
            if(n < 8192)
            {
                return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, 1, false, false, false, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 8, 16, 1, 1, false, false, false, false);
            }
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 2, 8, 16, 1, 1, false, false, false, false);
        }
    }
}
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute