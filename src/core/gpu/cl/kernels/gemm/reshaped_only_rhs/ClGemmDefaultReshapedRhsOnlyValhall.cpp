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
#include "src/core/gpu/cl/kernels/gemm/reshaped_only_rhs/ClGemmDefaultConfigReshapedRhsOnlyValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/gpu/cl/kernels/gemm/ClGemmHelpers.h"

#include <utility>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace gemm
{
using namespace arm_compute::misc::shape_calculator;

ClGemmDefaultConfigReshapedRhsOnlyValhall::ClGemmDefaultConfigReshapedRhsOnlyValhall(GPUTarget gpu)
    : IClGemmKernelConfig(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (ClGemmDefaultConfigReshapedRhsOnlyValhall::*)(unsigned int m, unsigned int n, unsigned int k,
                                             unsigned int b);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G77(&ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_f32,
                                                                    &ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_f16,
                                                                    &ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_u8);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G78(&ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G78_f32,
                                                                    &ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G78_f16,
                                                                    &ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_u8);

    ConfigurationFunctionExecutorPtr func = nullptr;

    switch(_target)
    {
        case GPUTarget::G78:
            func = configs_G78.get_function(data_type);
            break;
        case GPUTarget::G77:
        default:
            func = configs_G77.get_function(data_type);
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not support for GEMM");
    return (this->*func)(m, n, k, b);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    if(m == 1)
    {
        const float r_mn = static_cast<float>(m) / static_cast<float>(n);
        const float r_mk = static_cast<float>(m) / static_cast<float>(k);

        if(r_mk <= 0.0064484127797186375)
        {
            if(r_mn <= 0.0028273810748942196)
            {
                GEMMLHSMatrixInfo lhs_info_buf;
                GEMMRHSMatrixInfo rhs_info_buf;
                GEMMLHSMatrixInfo lhs_info_img;
                GEMMRHSMatrixInfo rhs_info_img;

                const unsigned int h0 = std::max(n / 4, 1U);
                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 1, 4, 8, 1, 16, 0, 1, 0, 0, 1);
                std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 1, 4, 4, 1, h0, 0, 1, 0, 1, 0);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf),
                                           n, k, b, DataType::F32);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 8, 0, 1, 0, 0, 0);
            }
        }
        else
        {
            if(r_mk <= 0.020312500186264515)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 4, 0, 1, 0, 0, 0);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, 16, 0, 1, 0, 1, 0);
            }
        }
    }
    else
    {
        const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
        const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
        const float r_mk     = static_cast<float>(m) / static_cast<float>(k);

        if(workload <= 1999.2000122070312)
        {
            if(workload <= 747.1999816894531)
            {
                return configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 8, 0, 1, 0, 1, 0);
            }
            else
            {
                GEMMLHSMatrixInfo lhs_info_buf;
                GEMMRHSMatrixInfo rhs_info_buf;
                GEMMLHSMatrixInfo lhs_info_img;
                GEMMRHSMatrixInfo rhs_info_img;
                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 2, 0, 0, 0, 1, 1);
                std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 8, 0, 1, 0, 1, 0);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf),
                                           n, k, b, DataType::F32);
            }
        }
        else
        {
            if(r_mn <= 0.03348214365541935)
            {
                if(r_mk <= 0.028125000186264515)
                {
                    return configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 8, 0, 1, 0, 1, 0);
                }
                else
                {
                    GEMMLHSMatrixInfo lhs_info_buf;
                    GEMMRHSMatrixInfo rhs_info_buf;
                    GEMMLHSMatrixInfo lhs_info_img;
                    GEMMRHSMatrixInfo rhs_info_img;
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 2, 0, 0, 0, 1, 1);
                    std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 8, 0, 1, 0, 1, 0);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F32);
                }
            }
            else
            {
                GEMMLHSMatrixInfo lhs_info_buf;
                GEMMRHSMatrixInfo rhs_info_buf;
                GEMMLHSMatrixInfo lhs_info_img;
                GEMMRHSMatrixInfo rhs_info_img;
                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 2, 0, 1, 0, 0, 1);
                std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 16, 0, 1, 0, 1, 0);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf),
                                           n, k, b, DataType::F32);
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int h0 = std::max(n / 2, 1U);
        if(n <= 836.0)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, h0, 0, 1, 0, 1, 0);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, h0, 0, 1, 0, 1, 0);
        }
    }
    else if(m < 128)
    {
        const int h0 = std::max(std::min(static_cast<int>(n / 4), static_cast<int>(256)), static_cast<int>(1));
        if(k >= 512)
        {
            return configure_lhs_rhs_info(m, n, 2, 4, 16, 1, h0, 0, 1, 0, 0);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, h0, 0, 1, 0, 0);
        }
    }
    else
    {
        const int h0 = std::max(std::min(static_cast<int>(n / 4), static_cast<int>(256)), static_cast<int>(1));
        if(n >= 64)
        {
            return configure_lhs_rhs_info(m, n, 4, 8, 4, 1, h0, 0, 1, 0, 0);
        }
        else
        {
            if(k >= 512)
            {
                return configure_lhs_rhs_info(m, n, 2, 4, 16, 1, h0, 0, 1, 0, 0);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, h0, 0, 1, 0, 0);
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int h0 = std::max(n / 2, 1U);
        return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, h0, 0, 1, 0, 1);
    }
    else
    {
        const int h0 = std::max(std::min(static_cast<int>(n / 4), static_cast<int>(256)), static_cast<int>(1));
        if(m >= 28)
        {
            return configure_lhs_rhs_info(m, n, 4, 4, 16, 1, h0, 0, 1, 0, 1);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 2, 4, 16, 1, h0, 0, 1, 0, 1);
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G78_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float r_mk     = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;

    if(m == 1)
    {
        if(workload <= 278.7000f)
        {
            if(workload <= 7.5000f)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, 2, 0, 1, 1, 0, 0);
            }
            else
            {
                if(r_mn <= 0.0031f)
                {
                    if(workload <= 256.6000f)
                    {
                        if(workload <= 16.7500f)
                        {
                            if(r_nk <= 1.6671f)
                            {
                                return configure_lhs_rhs_info(m, n, 1, 2, 2, 1, 32, 0, 0, 0, 1, 0);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, 2, 0, 1, 1, 0, 0);
                            }
                        }
                        else
                        {
                            return configure_lhs_rhs_info(m, n, 1, 2, 2, 1, 32, 0, 0, 0, 1, 0);
                        }
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 1, 2, 2, 1, 32, 0, 0, 0, 1, 0);
                    }
                }
                else
                {
                    if(r_mk <= 0.0027f)
                    {
                        if(r_mk <= 0.0014f)
                        {
                            return configure_lhs_rhs_info(m, n, 1, 2, 2, 1, 32, 0, 0, 0, 1, 0);
                        }
                        else
                        {
                            if(workload <= 8.9500f)
                            {
                                return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, 2, 0, 1, 1, 0, 0);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 1, 2, 2, 1, 32, 0, 0, 0, 1, 0);
                            }
                        }
                    }
                    else
                    {
                        if(workload <= 14.1500f)
                        {
                            return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, 2, 0, 1, 1, 0, 0);
                        }
                        else
                        {
                            if(r_mk <= 0.0041f)
                            {
                                return configure_lhs_rhs_info(m, n, 1, 2, 2, 1, 32, 0, 0, 0, 1, 0);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, 2, 0, 1, 1, 0, 0);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            if(workload <= 363.7000f)
            {
                if(r_mk <= 0.0031f)
                {
                    return configure_lhs_rhs_info(m, n, 1, 4, 2, 1, 32, 0, 1, 0, 1, 0);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 1, 4, 4, 1, 32, 0, 1, 0, 1, 0);
                }
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 4, 2, 1, 32, 0, 1, 0, 1, 0);
            }
        }
    }
    else
    {
        if(workload <= 1384.8000f)
        {
            if(workload <= 704.0000f)
            {
                return configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 32, 0, 1, 0, 1, 0);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 4, 0, 0, 0, 1, 1);
            }
        }
        else
        {
            if(workload <= 16761.6006f)
            {
                if(r_mn <= 187.1250f)
                {
                    return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 16, 0, 0, 0, 1, 1);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 4, 0, 0, 0, 1, 1);
                }
            }
            else
            {
                if(r_mk <= 432.4630f)
                {
                    return configure_lhs_rhs_info(m, n, 5, 4, 4, 1, 16, 0, 0, 0, 1, 1);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 16, 0, 1, 0, 1, 1);
                }
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G78_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float r_mk     = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;

    if(m == 1)
    {
        if(r_mn <= 0.0038f)
        {
            if(workload <= 353.9000f)
            {
                if(workload <= 278.7000f)
                {
                    return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 32, 0, 0, 1, 0, 0);
                }
                else
                {
                    if(r_mk <= 0.0004f)
                    {
                        return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 32, 0, 0, 1, 0, 0);
                    }
                    else
                    {
                        if(r_mk <= 0.0030f)
                        {
                            return configure_lhs_rhs_info(m, n, 1, 8, 4, 1, 8, 0, 1, 1, 0, 1);
                        }
                        else
                        {
                            return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 32, 0, 0, 1, 0, 0);
                        }
                    }
                }
            }
            else
            {
                if(r_nk <= 1.9384f)
                {
                    return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 32, 0, 0, 1, 0, 0);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 1, 8, 4, 1, 8, 0, 1, 1, 0, 1);
                }
            }
        }
        else
        {
            if(r_nk <= 1.0368f)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 32, 0, 0, 1, 0, 0);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, 32, 0, 0, 1, 0, 0);
            }
        }
    }
    else
    {
        if(workload <= 1422.4000f)
        {
            if(workload <= 704.0000f)
            {
                return configure_lhs_rhs_info(m, n, 2, 2, 8, 1, 32, 0, 0, 1, 0, 0);
            }
            else
            {
                if(workload <= 1197.6000f)
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 8, 0, 1, 1, 0, 1);
                }
                else
                {
                    if(workload <= 1241.6000f)
                    {
                        return configure_lhs_rhs_info(m, n, 2, 8, 8, 1, 16, 0, 1, 1, 0, 0);
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 8, 0, 1, 1, 0, 1);
                    }
                }
            }
        }
        else
        {
            if(workload <= 2769.6000f)
            {
                if(workload <= 1846.4000f)
                {
                    if(r_mn <= 2.4927f)
                    {
                        return configure_lhs_rhs_info(m, n, 2, 8, 8, 1, 16, 0, 1, 1, 0, 0);
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 32, 0, 1, 1, 0, 0);
                    }
                }
                else
                {
                    if(r_mn <= 0.6261f)
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 32, 0, 1, 1, 0, 0);
                    }
                    else
                    {
                        if(r_mk <= 3.4453f)
                        {
                            if(r_mn <= 1.4135f)
                            {
                                return configure_lhs_rhs_info(m, n, 2, 8, 8, 1, 16, 0, 1, 1, 0, 0);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 32, 0, 1, 1, 0, 0);
                            }
                        }
                        else
                        {
                            return configure_lhs_rhs_info(m, n, 2, 8, 8, 1, 16, 0, 1, 1, 0, 0);
                        }
                    }
                }
            }
            else
            {
                if(r_nk <= 0.0302f)
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 8, 0, 1, 1, 0, 1);
                }
                else
                {
                    if(r_mk <= 181.3750f)
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 32, 0, 1, 1, 0, 0);
                    }
                    else
                    {
                        if(workload <= 28035.2002f)
                        {
                            return configure_lhs_rhs_info(m, n, 2, 8, 8, 1, 16, 0, 1, 1, 0, 0);
                        }
                        else
                        {
                            if(r_mk <= 808.6667f)
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 32, 0, 1, 1, 0, 0);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 2, 8, 8, 1, 16, 0, 1, 1, 0, 0);
                            }
                        }
                    }
                }
            }
        }
    }
}
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
