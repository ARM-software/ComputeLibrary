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
#include "arm_compute/core/CL/gemm/native/CLGEMMNativeKernelConfigurationBifrost.h"

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
CLGEMMNativeKernelConfigurationBifrost::CLGEMMNativeKernelConfigurationBifrost(GPUTarget arch)
    : ICLGEMMKernelConfiguration(arch)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationBifrost::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    ARM_COMPUTE_ERROR_ON(data_type != DataType::F32 && data_type != DataType::QASYMM8);
    ARM_COMPUTE_UNUSED(data_type);

    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (CLGEMMNativeKernelConfigurationBifrost::*)(unsigned int m, unsigned int n, unsigned int k,
                                             unsigned int b);

    // Configurations for Mali-G71
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G71 =
    {
        { DataType::F32, &CLGEMMNativeKernelConfigurationBifrost::configure_G71_f32 },
        { DataType::QASYMM8, &CLGEMMNativeKernelConfigurationBifrost::configure_G71_u8 }
    };

    // Configurations for Mali-G76
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G76 =
    {
        { DataType::F32, &CLGEMMNativeKernelConfigurationBifrost::configure_G76_f32 },
        { DataType::QASYMM8, &CLGEMMNativeKernelConfigurationBifrost::configure_G76_u8 }
    };

    // Default configurations
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_default =
    {
        { DataType::F32, &CLGEMMNativeKernelConfigurationBifrost::configure_default_f32 },
        { DataType::QASYMM8, &CLGEMMNativeKernelConfigurationBifrost::configure_default_u8 }
    };

    switch(_target)
    {
        case GPUTarget::G71:
            return (this->*gemm_configs_G71[data_type])(m, n, k, b);
        case GPUTarget::G76:
            return (this->*gemm_configs_G76[data_type])(m, n, k, b);
        default:
            return (this->*gemm_configs_default[data_type])(m, n, k, b);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationBifrost::configure_G71_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationBifrost::configure_G71_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationBifrost::configure_G76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        if(n > 4196)
        {
            return configure_lhs_rhs_info(m, n, 1, 4, 2, 1, 1, false, false, false, false);
        }
        else
        {
            if(k < 2048)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 2, 1, 1, false, false, false, false);
            }
            else if(k >= 2048 && k < 16384)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 1, false, false, false, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, 1, false, false, false, false);
            }
        }
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 2, 8, 2, 1, 1, false, false, false, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationBifrost::configure_G76_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationBifrost::configure_default_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    return configure_lhs_rhs_info(m, n, 5, 4, 4, 1, 1, false, false, false, false);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationBifrost::configure_default_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    return configure_lhs_rhs_info(m, n, 5, 2, 16, 1, 1, false, false, false, false);
}
} // namespace cl_gemm
} // namespace arm_compute