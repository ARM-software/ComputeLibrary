/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/CL/gemm_reshaped/CLGEMMReshapedConfigurationBifrost.h"

#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
namespace
{
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_gemm_reshaped(unsigned int m, unsigned int n, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
                                                                        bool lhs_interleave, bool rhs_interleave)
{
    GEMMLHSMatrixInfo lhs_info;
    GEMMRHSMatrixInfo rhs_info;

    // Configure GEMMLHSMatrixInfo
    lhs_info.m0         = m0;
    lhs_info.k0         = k0;
    lhs_info.v0         = ((m / (lhs_info.m0 * v0)) == 0) ? 1 : v0;
    lhs_info.interleave = lhs_interleave;
    lhs_info.transpose  = false;

    // Configure GEMMRHSMatrixInfo
    rhs_info.n0         = n0;
    rhs_info.k0         = lhs_info.k0;
    rhs_info.h0         = ((n / (rhs_info.n0 * h0)) == 0) ? 1 : h0;
    rhs_info.interleave = rhs_interleave;
    rhs_info.transpose  = true;

    return std::make_pair(lhs_info, rhs_info);
}

} // namespace

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedConfigurationBifrost::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    ARM_COMPUTE_ERROR_ON(data_type != DataType::F32 && data_type != DataType::QASYMM8);
    ARM_COMPUTE_UNUSED(data_type);

    const GPUTarget gpu_target = CLScheduler::get().target();

    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (CLGEMMReshapedConfigurationBifrost::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b);

    // Configurations for Mali-G76
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_reshaped_configs_G76 =
    {
        { DataType::F32, &CLGEMMReshapedConfigurationBifrost::configure_G76_f32 },
        { DataType::QASYMM8, &CLGEMMReshapedConfigurationBifrost::configure_G76_u8 }
    };

    // Configurations for Mali-G7x
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_reshaped_configs_G7x =
    {
        { DataType::F32, &CLGEMMReshapedConfigurationBifrost::configure_G7x_f32 },
        { DataType::QASYMM8, &CLGEMMReshapedConfigurationBifrost::configure_G7x_u8 }
    };

    switch(gpu_target)
    {
        case GPUTarget::G76:
            return (this->*gemm_reshaped_configs_G76[data_type])(m, n, k, b);
        default:
            return (this->*gemm_reshaped_configs_G7x[data_type])(m, n, k, b);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedConfigurationBifrost::configure_G7x_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(n <= 4)
    {
        return configure_gemm_reshaped(m, n, 4, 2, 8, 16, 16, true, false);
    }
    else
    {
        return configure_gemm_reshaped(m, n, 5, 4, 4, 2, 16, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedConfigurationBifrost::configure_G7x_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(dot8_supported(CLKernelLibrary::get().get_device()))
    {
        if(n <= 4)
        {
            return configure_gemm_reshaped(m, n, 4, 2, 16, 2, 2, true, false);
        }
        else
        {
            return configure_gemm_reshaped(m, n, 4, 4, 16, 2, 2, true, false);
        }
    }
    else
    {
        if(n <= 4)
        {
            return configure_gemm_reshaped(m, n, 4, 2, 8, 2, 2, true, false);
        }
        else
        {
            return configure_gemm_reshaped(m, n, 6, 4, 4, 2, 2, true, true);
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedConfigurationBifrost::configure_G76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(n <= 4)
    {
        return configure_gemm_reshaped(m, n, 4, 2, 8, 16, 16, true, false);
    }
    else
    {
        return configure_gemm_reshaped(m, n, 4, 4, 2, 8, 16, false, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedConfigurationBifrost::configure_G76_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(n <= 4)
    {
        return configure_gemm_reshaped(m, n, 4, 2, 16, 4, 1, false, false);
    }
    else
    {
        return configure_gemm_reshaped(m, n, 4, 4, 16, 2, 2, false, true);
    }
}
} // namespace cl_gemm
} // namespace arm_compute