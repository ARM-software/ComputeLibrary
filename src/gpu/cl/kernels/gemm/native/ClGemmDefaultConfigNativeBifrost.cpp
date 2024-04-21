/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/gemm/native/ClGemmDefaultConfigNativeBifrost.h"

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
ClGemmDefaultConfigNativeBifrost::ClGemmDefaultConfigNativeBifrost(GPUTarget gpu) : IClGemmKernelConfig(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigNativeBifrost::configure(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (
        ClGemmDefaultConfigNativeBifrost::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G71(
        &ClGemmDefaultConfigNativeBifrost::configure_G71_f32,
        &ClGemmDefaultConfigNativeBifrost::configure_G71_f32, // We use the F32 heuristic
        &ClGemmDefaultConfigNativeBifrost::configure_G71_u8);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G76(
        &ClGemmDefaultConfigNativeBifrost::configure_G76_f32,
        &ClGemmDefaultConfigNativeBifrost::configure_G76_f32, // We use the F32 heuristic
        &ClGemmDefaultConfigNativeBifrost::configure_G76_u8);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G7x(
        &ClGemmDefaultConfigNativeBifrost::configure_default_f32,
        &ClGemmDefaultConfigNativeBifrost::configure_default_f32, // We use the F32 heuristic
        &ClGemmDefaultConfigNativeBifrost::configure_default_u8);

    ConfigurationFunctionExecutorPtr func = nullptr;

    switch (_target)
    {
        case GPUTarget::G76:
            func = configs_G76.get_function(data_type);
            break;
        case GPUTarget::G71:
            func = configs_G71.get_function(data_type);
            break;
        default:
            func = configs_G7x.get_function(data_type);
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not support for GEMM");
    return (this->*func)(m, n, k, b);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigNativeBifrost::configure_G71_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if (m == 1)
    {
        if (n < 2048)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 1, false, false, false, false);
        }
        else if (n >= 2048 && n < 8192)
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
        return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 1, false, false, false, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigNativeBifrost::configure_G71_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if (dot8_supported(CLKernelLibrary::get().get_device()))
    {
        if (m == 1)
        {
            if (n < 2048)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 1, false, false, false, false);
            }
            else if (n >= 2048 && n < 16384)
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
            if (m < 64)
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
        if (m == 1)
        {
            if (n < 8192)
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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigNativeBifrost::configure_G76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if (m == 1)
    {
        if (n > 4196)
        {
            return configure_lhs_rhs_info(m, n, 1, 4, 2, 1, 1, false, false, false, false);
        }
        else
        {
            if (k < 2048)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 2, 1, 1, false, false, false, false);
            }
            else if (k >= 2048 && k < 16384)
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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigNativeBifrost::configure_G76_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if (m == 1)
    {
        if (n < 2048)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 1, false, false, false, false);
        }
        else if (n >= 2048 && n < 16384)
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
        if (m < 64)
        {
            return configure_lhs_rhs_info(m, n, 2, 2, 16, 1, 1, false, false, false, false);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 5, 2, 16, 1, 1, false, false, false, false);
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigNativeBifrost::configure_default_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    return configure_lhs_rhs_info(m, n, 5, 4, 4, 1, 1, false, false, false, false);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigNativeBifrost::configure_default_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    return configure_lhs_rhs_info(m, n, 5, 2, 16, 1, 1, false, false, false, false);
}
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
