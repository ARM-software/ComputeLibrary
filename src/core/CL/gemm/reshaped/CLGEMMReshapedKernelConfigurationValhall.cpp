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
#include "src/core/CL/gemm/reshaped/CLGEMMReshapedKernelConfigurationValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"
#include "src/core/CL/gemm/CLGEMMHelpers.h"

#include <map>
#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
CLGEMMReshapedKernelConfigurationValhall::CLGEMMReshapedKernelConfigurationValhall(GPUTarget gpu)
    : ICLGEMMKernelConfiguration(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationValhall::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (CLGEMMReshapedKernelConfigurationValhall::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b);

    // Configurations for Mali-G77
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G77 =
    {
        { DataType::F32, &CLGEMMReshapedKernelConfigurationValhall::configure_G77_f32 },
        { DataType::F16, &CLGEMMReshapedKernelConfigurationValhall::configure_G77_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedKernelConfigurationValhall::configure_G77_u8 },
        { DataType::QSYMM8, &CLGEMMReshapedKernelConfigurationValhall::configure_G77_u8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMReshapedKernelConfigurationValhall::configure_G77_u8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMReshapedKernelConfigurationValhall::configure_G77_u8 }
    };

    switch(_target)
    {
        case GPUTarget::G77:
        default:
            if(gemm_configs_G77.find(data_type) != gemm_configs_G77.end())
            {
                return (this->*gemm_configs_G77[data_type])(m, n, k, b);
            }
            else
            {
                ARM_COMPUTE_ERROR("Not supported data type");
            }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationValhall::configure_G77_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(n <= 4)
    {
        return configure_lhs_rhs_info(m, n, 4, 2, 8, 16, 16, true, false, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 5, 4, 4, 2, 16, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationValhall::configure_G77_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
    const float r_mk = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk = static_cast<float>(n) / static_cast<float>(k);

    GEMMLHSMatrixInfo lhs_info_buf;
    GEMMRHSMatrixInfo rhs_info_buf;
    GEMMLHSMatrixInfo lhs_info_img;
    GEMMRHSMatrixInfo rhs_info_img;

    std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, false, false, true, false, false);

    if(r_mk <= 0.11824845522642136)
    {
        if(workload <= 880.0)
        {
            return configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 4, false, false, true, false, false);
        }
        else
        {
            if(r_nk <= 0.42521367967128754)
            {
                if(workload <= 1726.4000244140625)
                {
                    return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 2, false, false, true, false, false);
                }
                else
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, false, true, true, false, true);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
            }
            else
            {
                if(workload <= 1241.6000366210938)
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 4, false, false, true, false, false);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 4, false, false, true, false, false);
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
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, false, true, true, false, true);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 4, false, false, true, false, false);
                }
            }
            else
            {
                if(workload <= 2881.199951171875)
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, false, false, true, false, true);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
                else
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, false, true, true, false, true);

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
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, false, true, true, false, true);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
                else
                {
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, true, false, true, false, true);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
            }
            else
            {
                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, true, false, true, false, true);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                            std::make_pair(lhs_info_buf, rhs_info_buf),
                                            n, k, b, DataType::F16);
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationValhall::configure_G77_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(n <= 4)
    {
        return configure_lhs_rhs_info(m, n, 4, 2, 16, 4, 1, false, false, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 16, 2, 2, false, true, false, true);
    }
}
} // namespace cl_gemm
} // namespace arm_compute
