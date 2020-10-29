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
#include "src/core/CL/gemm/reshaped/CLGEMMReshapedKernelConfigurationBifrost.h"

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

CLGEMMReshapedKernelConfigurationBifrost::CLGEMMReshapedKernelConfigurationBifrost(GPUTarget gpu)
    : ICLGEMMKernelConfiguration(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationBifrost::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (CLGEMMReshapedKernelConfigurationBifrost::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b);

    // Configurations for Mali-G76
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G76 =
    {
        { DataType::F32, &CLGEMMReshapedKernelConfigurationBifrost::configure_G76_f32 },
        { DataType::F16, &CLGEMMReshapedKernelConfigurationBifrost::configure_G76_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedKernelConfigurationBifrost::configure_G76_u8 },
        { DataType::QSYMM8, &CLGEMMReshapedKernelConfigurationBifrost::configure_G76_u8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMReshapedKernelConfigurationBifrost::configure_G76_u8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMReshapedKernelConfigurationBifrost::configure_G76_u8 }
    };

    // Configurations for Mali-G7x
    static std::map<DataType, ConfigurationFunctionExecutorPtr> gemm_configs_G7x =
    {
        { DataType::F32, &CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_f32 },
        { DataType::F16, &CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_f16 },
        { DataType::QASYMM8, &CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QSYMM8, &CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_u8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_u8 }
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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
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

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(n <= 4)
    {
        return configure_lhs_rhs_info(m, n, 4, 2, 8, 8, 2, true, true, true, false);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 8, 4, 4, 2, true, true, true, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationBifrost::configure_G7x_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if(dot8_supported(CLKernelLibrary::get().get_device()))
    {
        if(n <= 4)
        {
            return configure_lhs_rhs_info(m, n, 4, 2, 16, 2, 2, true, false, false, true);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 4, 4, 16, 2, 2, true, false, false, true);
        }
    }
    else
    {
        if(n <= 4)
        {
            return configure_lhs_rhs_info(m, n, 4, 2, 8, 2, 2, true, false, false, true);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 6, 4, 4, 2, 2, true, true, false, true);
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationBifrost::configure_G76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    GEMMLHSMatrixInfo lhs_info_buf;
    GEMMRHSMatrixInfo rhs_info_buf;
    GEMMLHSMatrixInfo lhs_info_img;
    GEMMRHSMatrixInfo rhs_info_img;

    // Get lhs_info/rhs_info in case of OpenCL buffer
    if(n <= 4)
    {
        std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 2, 8, 16, 16, true, false, false, true);
    }
    else
    {
        std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 4, 2, 8, 16, false, false, false, true);
    }

    // Get lhs_info/rhs_info in case of OpenCL image
    // Condition on the GPU workload
    if((m / 4) * (n / 4) >= 2560)
    {
        // Big workload
        std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 8, true, true, true, false, true);
    }
    else
    {
        // Small workload
        std::tie(lhs_info_img, rhs_info_img) = configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 1, true, true, true, false, true);
    }

    const TensorInfo  tensor_rhs_info(TensorShape(n, k, b), 1, DataType::F32);
    const TensorShape shape = compute_rhs_reshaped_shape(tensor_rhs_info, rhs_info_img);
    const TensorInfo  tensor_reshaped_info(shape, 1, DataType::F32);

    // In case of vector by matrix with few work-items, we use the OpenCL buffer rather than the OpenCL image2d
    const bool use_cl_image2d = (n <= 4) ? false : true;

    if(bool(validate_image2d_support_on_rhs(tensor_reshaped_info, rhs_info_img)) && use_cl_image2d)
    {
        return std::make_pair(lhs_info_img, rhs_info_img);
    }
    else
    {
        return std::make_pair(lhs_info_buf, rhs_info_buf);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationBifrost::configure_G76_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
    const float r_mk = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk = static_cast<float>(n) / static_cast<float>(k);

    if(workload <= 1422.40f)
    {
        if(r_mk <= 2.45f)
        {
            if(workload <= 801.60f)
            {
                return configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 2, true, false, true, false, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 4, 2, 4, 2, 2, false, false, true, false, false);
            }
        }
        else
        {
            if(r_nk <= 0.67f)
            {
                return configure_lhs_rhs_info(m, n, 4, 2, 4, 2, 2, false, false, true, false, false);
            }
            else
            {
                return configure_lhs_rhs_info(m, n, 2, 4, 4, 4, 1, false, true, false, true, false);
            }
        }
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 4, true, true, true, false, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedKernelConfigurationBifrost::configure_G76_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
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
