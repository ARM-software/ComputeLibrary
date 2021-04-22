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
#include "src/core/gpu/cl/kernels/gemm/reshaped/ClGemmDefaultConfigReshapedValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"
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
ClGemmDefaultConfigReshapedValhall::ClGemmDefaultConfigReshapedValhall(GPUTarget gpu)
    : IClGemmKernelConfig(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedValhall::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (ClGemmDefaultConfigReshapedValhall::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G77(&ClGemmDefaultConfigReshapedValhall::configure_G77_f32,
                                                                    &ClGemmDefaultConfigReshapedValhall::configure_G77_f16,
                                                                    &ClGemmDefaultConfigReshapedValhall::configure_G77_u8);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G78(&ClGemmDefaultConfigReshapedValhall::configure_G78_f32,
                                                                    &ClGemmDefaultConfigReshapedValhall::configure_G78_f16,
                                                                    &ClGemmDefaultConfigReshapedValhall::configure_G77_u8);

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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedValhall::configure_G77_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(n <= 4)
    {
        return configure_lhs_rhs_info(m, n, 4, 2, 8, 16, 16, 1, 0, 0, 1);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 5, 4, 4, 2, 16, 0, 1, 0, 1);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedValhall::configure_G77_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
    const float r_mk     = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);

    GEMMLHSMatrixInfo lhs_info_buf;
    GEMMRHSMatrixInfo rhs_info_buf;
    GEMMLHSMatrixInfo lhs_info_img;
    GEMMRHSMatrixInfo rhs_info_img;

    std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 1, 0, 0);

    if(r_mk <= 0.11824845522642136)
    {
        if(workload <= 880.0)
        {
            return configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 4, 0, 0, 1, 0, 0);
        }
        else
        {
            if(r_nk <= 0.42521367967128754)
            {
                if(workload <= 1726.4000244140625)
                {
                    return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 0);
                }
                else
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, 0, 1, 1, 0, 1);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
            }
            else
            {
                if(workload <= 1241.6000366210938)
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 4, 0, 0, 1, 0, 0);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 1, 0, 0);
                }
            }
        }
    }
    else
    {
        if(workload <= 11404.7998046875)
        {
            if(r_mk <= 1.0126488208770752)
            {
                if(r_mn <= 2.545312523841858)
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, 0, 1, 1, 0, 1);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 4, 0, 0, 1, 0, 0);
                }
            }
            else
            {
                if(workload <= 2881.199951171875)
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, 0, 0, 1, 0, 1);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
                else
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, 0, 1, 1, 0, 1);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
            }
        }
        else
        {
            if(r_nk <= 0.5765306055545807)
            {
                if(r_mn <= 6.010416746139526)
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, 0, 1, 1, 0, 1);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
                else
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, 1, 0, 1, 0, 1);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
            }
            else
            {
                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, 1, 0, 1, 0, 1);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf),
                                           n, k, b, DataType::F16);
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedValhall::configure_G78_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float r_mk     = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;

    if(workload <= 1288.0000f)
    {
        if(workload <= 505.6000f)
        {
            if(r_mn <= 0.4466f)
            {
                if(r_nk <= 0.2384f)
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 8, 4, 4, 0, 0, 1, 0, 1);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 2, 2, 4, 2, 2, 0, 0, 1, 0, 0);
                }
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 2, 2, 4, 2, 2, 0, 0, 1, 0, 0);
            }
        }
        else
        {
            if(r_mn <= 0.2250f)
            {
                if(r_mn <= 0.1599f)
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 8, 4, 4, 0, 0, 1, 0, 1);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                }
            }
            else
            {
                if(r_mk <= 0.7609f)
                {
                    if(r_mn <= 2.5453f)
                    {
                        if(workload <= 1089.6000f)
                        {
                            return configure_lhs_rhs_info(m, n, 2, 4, 8, 4, 4, 0, 0, 1, 0, 1);
                        }
                        else
                        {
                            return configure_lhs_rhs_info(m, n, 2, 4, 8, 2, 4, 0, 0, 1, 0, 1);
                        }
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 2, 4, 16, 4, 4, 0, 0, 1, 0, 1);
                    }
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 8, 4, 4, 0, 0, 1, 0, 1);
                }
            }
        }
    }
    else
    {
        if(workload <= 5434.4001f)
        {
            if(workload <= 1603.2000f)
            {
                return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
            }
            else
            {
                if(r_nk <= 0.6192f)
                {
                    if(r_mn <= 16.1016f)
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                    }
                    else
                    {
                        if(workload <= 2750.0000f)
                        {
                            return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                        }
                        else
                        {
                            if(r_mk <= 6.3151f)
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                            }
                        }
                    }
                }
                else
                {
                    if(r_mk <= 0.0387f)
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 1, 0, 1);
                    }
                    else
                    {
                        if(r_mk <= 2.5859f)
                        {
                            if(r_mk <= 0.2734f)
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 1, 0, 1);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                            }
                        }
                        else
                        {
                            return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                        }
                    }
                }
            }
        }
        else
        {
            if(r_mk <= 25.7500f)
            {
                if(r_mk <= 0.3615f)
                {
                    if(r_mn <= 0.0913f)
                    {
                        if(r_mk <= 0.0683f)
                        {
                            return configure_lhs_rhs_info(m, n, 8, 4, 4, 4, 2, 0, 0, 1, 0, 1);
                        }
                        else
                        {
                            return configure_lhs_rhs_info(m, n, 2, 4, 8, 4, 4, 0, 0, 1, 0, 1);
                        }
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 8, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                    }
                }
                else
                {
                    if(workload <= 11174.3999f)
                    {
                        if(r_mk <= 0.8047f)
                        {
                            return configure_lhs_rhs_info(m, n, 8, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                        }
                        else
                        {
                            if(workload <= 7185.5999f)
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 1, 0, 1);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 8, 4, 4, 4, 2, 0, 0, 1, 0, 1);
                            }
                        }
                    }
                    else
                    {
                        if(workload <= 17917.5000f)
                        {
                            if(r_mk <= 1.5078f)
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 1, 0, 1);
                            }
                        }
                        else
                        {
                            if(workload <= 34449.6016f)
                            {
                                return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                            }
                            else
                            {
                                return configure_lhs_rhs_info(m, n, 8, 4, 4, 2, 4, 0, 0, 1, 0, 1);
                            }
                        }
                    }
                }
            }
            else
            {
                if(r_mk <= 331.1111f)
                {
                    if(workload <= 53397.5996f)
                    {
                        if(r_mn <= 57.8063f)
                        {
                            return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                        }
                        else
                        {
                            return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1);
                        }
                    }
                    else
                    {
                        if(r_nk <= 0.9211f)
                        {
                            return configure_lhs_rhs_info(m, n, 8, 4, 4, 4, 2, 0, 0, 1, 0, 1);
                        }
                        else
                        {
                            return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1);
                        }
                    }
                }
                else
                {
                    if(workload <= 38070.4004f)
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1);
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                    }
                }
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedValhall::configure_G78_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;

    if(workload <= 801.6000f)
    {
        return configure_lhs_rhs_info(m, n, 8, 4, 4, 1, 1, 0, 0, 1, 0, 1);
    }
    else
    {
        if(r_mn <= 0.1211f)
        {
            if(workload <= 3296.0000f)
            {
                return configure_lhs_rhs_info(m, n, 8, 4, 4, 2, 2, 0, 0, 1, 0, 1);
            }
            else
            {
                if(r_nk <= 1.0625f)
                {
                    return configure_lhs_rhs_info(m, n, 8, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 8, 4, 4, 2, 4, 0, 0, 1, 0, 1);
                }
            }
        }
        else
        {
            if(workload <= 5068.8000f)
            {
                return configure_lhs_rhs_info(m, n, 8, 4, 4, 1, 1, 0, 0, 1, 0, 1);
            }
            else
            {
                if(r_nk <= 0.2361f)
                {
                    if(workload <= 12630.0000f)
                    {
                        return configure_lhs_rhs_info(m, n, 8, 4, 4, 1, 1, 0, 0, 1, 0, 1);
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 8, 4, 4, 2, 1, 0, 0, 1, 0, 1);
                    }
                }
                else
                {
                    if(workload <= 178790.3984f)
                    {
                        return configure_lhs_rhs_info(m, n, 8, 4, 4, 2, 2, 0, 0, 1, 0, 1);
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 8, 4, 4, 1, 1, 0, 0, 1, 0, 1);
                    }
                }
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedValhall::configure_G77_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(n <= 4)
    {
        return configure_lhs_rhs_info(m, n, 4, 2, 16, 4, 1, 0, 0, 0, 1);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 16, 2, 2, 0, 1, 0, 1);
    }
}
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
