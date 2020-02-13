/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/core/CL/gemm/reshaped_only_rhs/CLGEMMReshapedOnlyRHSKernelConfigurationBifrost.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/gemm/CLGEMMHelpers.h"
#include "arm_compute/core/GPUTarget.h"

#include <map>
#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::CLGEMMReshapedOnlyRHSKernelConfigurationBifrost(GPUTarget gpu)
    : ICLGEMMKernelConfiguration(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::*)(unsigned int m, unsigned int n, unsigned int k,
                                             unsigned int b);

    // Configurations for Mali-G51
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G51 =
    {
        { DataType::F32, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_f32 },
        { DataType::F16, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_u8 }
    };

    // Configurations for Mali-G76
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G76 =
    {
        { DataType::F32, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_f32 },
        { DataType::F16, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_u8 }
    };

    // Configurations for Mali-G7x
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G7x =
    {
        { DataType::F32, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_f32 },
        { DataType::F16, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 }
    };

    switch(_target)
    {
        case GPUTarget::G76:
            if(gemm_configs_G76.find(data_type) != gemm_configs_G76.end())
            {
                return (this->*gemm_configs_G76[data_type])(m, n, k, b);
            }
            else
            {
                ARM_COMPUTE_ERROR("Not supported data type");
            }
        case GPUTarget::G51:
            if(gemm_configs_G51.find(data_type) != gemm_configs_G51.end())
            {
                return (this->*gemm_configs_G51[data_type])(m, n, k, b);
            }
            else
            {
                ARM_COMPUTE_ERROR("Not supported data type");
            }
        default:
            if(gemm_configs_G7x.find(data_type) != gemm_configs_G7x.end())
            {
                return (this->*gemm_configs_G7x[data_type])(m, n, k, b);
            }
            else
            {
                ARM_COMPUTE_ERROR("Not supported data type");
            }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        if(n > 2048)
        {
            const unsigned int h0 = std::max(n / 4, 1U);
            return configure_lhs_rhs_info(m, n, 1, 4, 4, 1, h0, false, true, false, true);
        }
        else
        {
            const unsigned int h0 = std::max(n / 2, 1U);
            return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, h0, false, true, false, true);
        }
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 4, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int h0 = std::max(n / 2, 1U);
        return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, h0, false, true, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 2, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int n0 = n < 1280 ? 2 : 4;
        const unsigned int h0 = std::max(n / n0, 1U);
        return configure_lhs_rhs_info(m, n, 1, n0, 4, 1, h0, false, true, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 2, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        if(n > 2048)
        {
            const unsigned int h0 = std::max(n / 4, 1U);
            return configure_lhs_rhs_info(m, n, 1, 4, 4, 1, h0, false, true, false, true);
        }
        else
        {
            const unsigned int h0 = std::max(n / 2, 1U);
            return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, h0, false, true, false, true);
        }
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 4, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int h0 = std::max(n / 2, 1U);
        return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, h0, false, true, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 2, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int n0 = n < 1280 ? 2 : 4;
        const unsigned int h0 = std::max(n / n0, 1U);
        return configure_lhs_rhs_info(m, n, 1, n0, 8, 1, h0, false, true, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 2, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(dot8_supported(CLKernelLibrary::get().get_device()))
    {
        if(m == 1)
        {
            const unsigned int h0 = std::max(n / 2, 1U);
            return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, h0, false, true, false, true);
        }
        else
        {
            const unsigned int h0 = std::max(n / 4, 1U);
            return configure_lhs_rhs_info(m, n, 4, 4, 16, 1, h0, false, true, false, true);
        }
    }
    else
    {
        if(m == 1)
        {
            const unsigned int h0 = std::max(n / 2, 1U);
            return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, h0, false, true, false, true);
        }
        else
        {
            const unsigned int h0 = std::max(n / 4, 1U);
            return configure_lhs_rhs_info(m, n, 2, 2, 16, 1, h0, false, true, false, true);
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int h0 = std::max(n / 2, 1U);
        return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, h0, false, true, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 16, 1, 2, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int h0 = std::max(n / 2, 1U);
        return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, h0, false, true, false, true);
    }
    else
    {
        const unsigned int h0 = std::max(n / 2, 1U);
        return configure_lhs_rhs_info(m, n, 4, 2, 16, 1, h0, false, true, false, true);
    }
}

} // namespace cl_gemm
} // namespace arm_compute
