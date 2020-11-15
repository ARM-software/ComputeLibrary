/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "src/core/CL/gemm/reshaped_only_rhs/CLGEMMReshapedOnlyRHSKernelConfigurationBifrost.h"

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
        { DataType::QASYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_u8 },
        { DataType::QSYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_u8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_u8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G51_u8 }
    };

    // Configurations for Mali-G52
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G52 =
    {
        { DataType::F32, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G52_f32 },
        { DataType::F16, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G52_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QSYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 }
    };

    // Configurations for Mali-G76
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G76 =
    {
        { DataType::F32, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_f32 },
        { DataType::F16, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_u8 },
        { DataType::QSYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_u8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_u8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_u8 }
    };

    // Configurations for Mali-G7x
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G7x =
    {
        { DataType::F32, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_f32 },
        { DataType::F16, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QSYMM8, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G7x_u8 }
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
        case GPUTarget::G52:
            if(gemm_configs_G52.find(data_type) != gemm_configs_G52.end())
            {
                return (this->*gemm_configs_G52[data_type])(m, n, k, b);
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
        if(n <= 2548)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 4, false, true, false, true, false);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, 8, false, true, false, true, false);
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

    GEMMLHSMatrixInfo lhs_info_buf;
    GEMMRHSMatrixInfo rhs_info_buf;
    GEMMLHSMatrixInfo lhs_info_img;
    GEMMRHSMatrixInfo rhs_info_img;

    const bool is_workload_big = ((m * n * b) / 16) >= 2048;

    if(m == 1)
    {
        if(n >= 8192)
        {
            const unsigned int h0 = std::max(n / 4, 1U);
            return configure_lhs_rhs_info(m, n, 1, 4, 8, 1, h0, false, true, false, true, false);
        }
        else
        {
            const unsigned int h0 = std::max(n / 2, 1U);
            if(n <= 204)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, h0, false, true, false, true, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 8, 1, h0, false, true, false, true, false);
            }
        }
    }
    else
    {
        const int h0 = std::max(std::min(static_cast<int>(n / 4), static_cast<int>(16)), static_cast<int>(1));
        if(is_workload_big)
        {
            std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 4, 4, 1, h0, false, true, false, true);
        }
        else
        {
            std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 2, 4, 8, 1, h0, false, true, false, true);
        }
    }

    // Get lhs_info/rhs_info in case of OpenCL image
    const int h0 = std::max(std::min(static_cast<int>(n / 4), static_cast<int>(16)), static_cast<int>(1));
    if(is_workload_big)
    {
        std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 1, h0, false, true, false, false, true);
    }
    else
    {
        std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 2, 4, 8, 1, h0, false, true, false, true, true);
    }

    const TensorInfo  tensor_rhs_info(TensorShape(n, k, b), 1, DataType::F32);
    const TensorShape shape = compute_rhs_reshaped_shape(tensor_rhs_info, rhs_info_img);
    const TensorInfo  tensor_reshaped_info(shape, 1, DataType::F32);

    // In case of vector by matrix or small workloads, we use the OpenCL buffer rather than the OpenCL image2d
    const bool use_cl_image2d = ((m == 1) || ((((m * n * b) / 16) < 2048) && n < 128)) ? false : true;

    if(bool(validate_image2d_support_on_rhs(tensor_reshaped_info, rhs_info_img)) && use_cl_image2d)
    {
        return std::make_pair(lhs_info_img, rhs_info_img);
    }
    else
    {
        return std::make_pair(lhs_info_buf, rhs_info_buf);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G52_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);

    GEMMLHSMatrixInfo lhs_info_buf;
    GEMMRHSMatrixInfo rhs_info_buf;
    GEMMLHSMatrixInfo lhs_info_img;
    GEMMRHSMatrixInfo rhs_info_img;

    if(m == 1)
    {
        if(r_nk <= 0.4664f)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 16, false, true, false, true, false);
        }
        else
        {
            std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 1, 4, 8, 1, 16, false, true, false, true, true);
            std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 1, 4, 8, 1, 16, false, true, false, true, false);

            return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                       std::make_pair(lhs_info_buf, rhs_info_buf),
                                       n, k, b, DataType::F32);
        }
    }
    else
    {
        if(workload <= 274.4000f)
        {
            return configure_lhs_rhs_info(m, n, 2, 2, 4, 1, 16, false, false, false, true, false);
        }
        else
        {
            std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 2, false, false, false, true, true);
            std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 4, 4, 1, 2, false, false, false, true, false);

            return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                       std::make_pair(lhs_info_buf, rhs_info_buf),
                                       n, k, b, DataType::F32);
        }
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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G52_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
    const float r_mk = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk = static_cast<float>(n) / static_cast<float>(k);

    if(m == 1)
    {
        if(r_mk <= 0.0026f)
        {
            if(r_nk <= 0.4664f)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 32, false, true, false, true, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, 16, false, true, false, false, true);
            }
        }
        else
        {
            if(r_mk <= 0.0148f)
            {
                return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 32, false, true, false, true, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 1, 4, 16, 1, 16, false, true, false, false, true);
            }
        }
    }
    else
    {
        if(workload <= 362.6000f)
        {
            return configure_lhs_rhs_info(m, n, 2, 2, 8, 1, 16, false, false, false, true, false);
        }
        else
        {
            if(r_mn <= 22.6067f)
            {
                if(workload <= 708.8000f)
                {
                    return configure_lhs_rhs_info(m, n, 5, 4, 4, 1, 2, false, false, false, false, true);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 5, 8, 2, 1, 16, false, false, false, false, false);
                }
            }
            else
            {
                if(r_nk <= 0.0917f)
                {
                    return configure_lhs_rhs_info(m, n, 2, 2, 8, 1, 16, false, false, false, true, false);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 5, 4, 4, 1, 2, false, false, false, false, true);
                }
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedOnlyRHSKernelConfigurationBifrost::configure_G76_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);

    if(m == 1)
    {
        return configure_lhs_rhs_info(m, n, 1, 2, 16, 1, 32, false, true, false, true, false);
    }
    else
    {
        const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
        const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;

        if(workload <= 7449.60f)
        {
            if(workload <= 691.60f)
            {
                return configure_lhs_rhs_info(m, n, 2, 2, 8, 1, 8, false, false, false, false, false);
            }
            else
            {
                if(workload <= 4155.20f)
                {
                    return configure_lhs_rhs_info(m, n, 5, 2, 8, 1, 16, false, false, false, false, false);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 5, 8, 2, 1, 32, false, false, false, false, false);
                }
            }
        }
        else
        {
            if(workload <= 16300.80f)
            {
                if(r_mn <= 44.56f)
                {
                    GEMMLHSMatrixInfo lhs_info_buf;
                    GEMMRHSMatrixInfo rhs_info_buf;
                    GEMMLHSMatrixInfo lhs_info_img;
                    GEMMRHSMatrixInfo rhs_info_img;

                    std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 5, 4, 4, 1, 2, false, true, false, false, true);
                    std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 5, 2, 8, 1, 16, false, false, false, false, false);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf),
                                               n, k, b, DataType::F16);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 5, 2, 8, 1, 16, false, false, false, false, false);
                }
            }
            else
            {
                GEMMLHSMatrixInfo lhs_info_buf;
                GEMMRHSMatrixInfo rhs_info_buf;
                GEMMLHSMatrixInfo lhs_info_img;
                GEMMRHSMatrixInfo rhs_info_img;

                std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 5, 4, 4, 1, 2, false, true, false, false, true);
                std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 5, 2, 8, 1, 16, false, false, false, false, false);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf),
                                           n, k, b, DataType::F16);
            }
        }
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
        const int h0 = std::max(std::min(static_cast<int>(n / 2), static_cast<int>(128)), static_cast<int>(1));
        if(m == 1)
        {
            return configure_lhs_rhs_info(m, n, 1, 2, 4, 1, h0, false, true, false, true);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 4, 2, 16, 1, h0, false, true, false, true);
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
