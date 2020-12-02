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
#include "src/core/CL/gemm/reshaped_only_rhs/CLGEMMDefaultConfigReshapedRHSOnlyValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/CL/gemm/CLGEMMHelpers.h"

#include <map>
#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
using namespace arm_compute::misc::shape_calculator;

CLGEMMDefaultConfigReshapedRHSOnlyValhall::CLGEMMDefaultConfigReshapedRHSOnlyValhall(GPUTarget gpu)
    : ICLGEMMKernelConfiguration(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (CLGEMMDefaultConfigReshapedRHSOnlyValhall::*)(unsigned int m, unsigned int n, unsigned int k,
                                             unsigned int b);

    // Configurations for Mali-G77
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G77 =
    {
        { DataType::F32, &CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_f32 },
        { DataType::F16, &CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_f16 },
        { DataType::QASYMM8, &CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_u8 },
        { DataType::QSYMM8, &CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_u8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_u8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_u8 }
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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
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
                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 1, 4, 8, 1, 16, false, true, false, false, true);
                std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 1, 4, 4, 1, h0, false, true, false, true, false);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf),
                                           n, k, b, DataType::F32);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 8, false, true, false, false, false);
            }
        }
        else
        {
            if(r_mk <= 0.020312500186264515)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 4, false, true, false, false, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, 16, false, true, false, true, false);
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
                return configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 8, false, true, false, true, false);
            }
            else
            {
                GEMMLHSMatrixInfo lhs_info_buf;
                GEMMRHSMatrixInfo rhs_info_buf;
                GEMMLHSMatrixInfo lhs_info_img;
                GEMMRHSMatrixInfo rhs_info_img;
                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 2, false, false, false, true, true);
                std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 8, false, true, false, true, false);

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
                    return configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 8, false, true, false, true, false);
                }
                else
                {
                    GEMMLHSMatrixInfo lhs_info_buf;
                    GEMMRHSMatrixInfo rhs_info_buf;
                    GEMMLHSMatrixInfo lhs_info_img;
                    GEMMRHSMatrixInfo rhs_info_img;
                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 2, 4, 8, 1, 2, false, false, false, true, true);
                    std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 8, false, true, false, true, false);

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
                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 2, false, true, false, false, true);
                std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 16, false, true, false, true, false);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf),
                                           n, k, b, DataType::F32);
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(m == 1)
    {
        const unsigned int h0 = std::max(n / 2, 1U);
        if(n <= 836.0)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, h0, false, true, false, true, false);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, h0, false, true, false, true, false);
        }
    }
    else if(m < 128)
    {
        const int h0 = std::max(std::min(static_cast<int>(n / 4), static_cast<int>(256)), static_cast<int>(1));
        if(k >= 512)
        {
            return configure_lhs_rhs_info(m, n, 2, 4, 16, 1, h0, false, true, false, false);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, h0, false, true, false, false);
        }
    }
    else
    {
        const int h0 = std::max(std::min(static_cast<int>(n / 4), static_cast<int>(256)), static_cast<int>(1));
        if(n >= 64)
        {
            return configure_lhs_rhs_info(m, n, 4, 4, 4, 1, h0, false, true, false, false);
        }
        else
        {
            if(k >= 512)
            {
                return configure_lhs_rhs_info(m, n, 2, 4, 16, 1, h0, false, true, false, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 2, 4, 8, 1, h0, false, true, false, false);
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMDefaultConfigReshapedRHSOnlyValhall::configure_G77_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
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
        const int h0 = std::max(std::min(static_cast<int>(n / 4), static_cast<int>(256)), static_cast<int>(1));
        if(m >= 28)
        {
            return configure_lhs_rhs_info(m, n, 4, 4, 16, 1, h0, false, true, false, true);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 2, 4, 16, 1, h0, false, true, false, true);
        }
    }
}
} // namespace cl_gemm
} // namespace arm_compute
