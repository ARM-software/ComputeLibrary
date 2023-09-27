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
#include "src/gpu/cl/kernels/gemm/reshaped/ClGemmDefaultConfigReshapedBifrost.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

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
using namespace arm_compute::misc::shape_calculator;

ClGemmDefaultConfigReshapedBifrost::ClGemmDefaultConfigReshapedBifrost(GPUTarget gpu) : IClGemmKernelConfig(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedBifrost::configure(
    unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (
        ClGemmDefaultConfigReshapedBifrost::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G7x(
        &ClGemmDefaultConfigReshapedBifrost::configure_G7x_f32, &ClGemmDefaultConfigReshapedBifrost::configure_G7x_f16,
        &ClGemmDefaultConfigReshapedBifrost::configure_G7x_u8);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G52(
        &ClGemmDefaultConfigReshapedBifrost::configure_G52_f32, &ClGemmDefaultConfigReshapedBifrost::configure_G52_f16,
        &ClGemmDefaultConfigReshapedBifrost::configure_G7x_u8);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G76(
        &ClGemmDefaultConfigReshapedBifrost::configure_G76_f32, &ClGemmDefaultConfigReshapedBifrost::configure_G76_f16,
        &ClGemmDefaultConfigReshapedBifrost::configure_G76_u8);

    ConfigurationFunctionExecutorPtr func = nullptr;

    switch (_target)
    {
        case GPUTarget::G76:
            func = configs_G76.get_function(data_type);
            break;
        case GPUTarget::G52:
            func = configs_G52.get_function(data_type);
            break;
        default:
            func = configs_G7x.get_function(data_type);
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not support for GEMM");
    return (this->*func)(m, n, k, b);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigReshapedBifrost::configure_G7x_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if (n <= 4)
    {
        return configure_lhs_rhs_info(m, n, 4, 2, 8, 16, 16, true, false, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 5, 4, 4, 2, 16, false, true, false, true);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigReshapedBifrost::configure_G7x_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if (n <= 4)
    {
        return configure_lhs_rhs_info(m, n, 4, 2, 8, 8, 2, true, true, true, false);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 8, 4, 4, 2, true, true, true, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigReshapedBifrost::configure_G7x_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if (dot8_supported(CLKernelLibrary::get().get_device()))
    {
        if (n <= 4)
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
        if (n <= 4)
        {
            return configure_lhs_rhs_info(m, n, 4, 2, 8, 2, 2, true, false, false, true);
        }
        else
        {
            return configure_lhs_rhs_info(m, n, 6, 4, 4, 2, 2, true, true, false, true);
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigReshapedBifrost::configure_G52_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
    const float r_mk     = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);

    GEMMLHSMatrixInfo lhs_info_buf;
    GEMMRHSMatrixInfo rhs_info_buf;
    GEMMLHSMatrixInfo lhs_info_img;
    GEMMRHSMatrixInfo rhs_info_img;

    if (workload <= 274.4000f)
    {
        if (r_nk <= 0.7461f)
        {
            if (r_mn <= 21.1667f)
            {
                return configure_lhs_rhs_info(m, n, 4, 2, 4, 4, 4, false, true, true, false, false);
            }
            else
            {
                std::tie(lhs_info_img, rhs_info_img) =
                    configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, true);
                std::tie(lhs_info_buf, rhs_info_buf) =
                    configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, false);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf), n, k, b, DataType::F32);
            }
        }
        else
        {
            std::tie(lhs_info_img, rhs_info_img) =
                configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, true);
            std::tie(lhs_info_buf, rhs_info_buf) =
                configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, false);

            return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                       std::make_pair(lhs_info_buf, rhs_info_buf), n, k, b, DataType::F32);
        }
    }
    else
    {
        if (r_mk <= 17.3926f)
        {
            if (workload <= 542.4000f)
            {
                std::tie(lhs_info_img, rhs_info_img) =
                    configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, true);
                std::tie(lhs_info_buf, rhs_info_buf) =
                    configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, false);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf), n, k, b, DataType::F32);
            }
            else
            {
                std::tie(lhs_info_img, rhs_info_img) =
                    configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, true, true, false, true, true);
                std::tie(lhs_info_buf, rhs_info_buf) =
                    configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, true, true, false, true, false);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf), n, k, b, DataType::F32);
            }
        }
        else
        {
            if (r_nk <= 0.5463f)
            {
                if (workload <= 11767.6001f)
                {
                    std::tie(lhs_info_img, rhs_info_img) =
                        configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, true);
                    std::tie(lhs_info_buf, rhs_info_buf) =
                        configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, false);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf), n, k, b, DataType::F32);
                }
                else
                {
                    std::tie(lhs_info_img, rhs_info_img) =
                        configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, true, true, false, true, true);
                    std::tie(lhs_info_buf, rhs_info_buf) =
                        configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 1, true, true, false, true, false);

                    return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                               std::make_pair(lhs_info_buf, rhs_info_buf), n, k, b, DataType::F32);
                }
            }
            else
            {
                std::tie(lhs_info_img, rhs_info_img) =
                    configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, true);
                std::tie(lhs_info_buf, rhs_info_buf) =
                    configure_lhs_rhs_info(m, n, 4, 4, 4, 4, 2, true, true, false, true, false);

                return select_lhs_rhs_info(std::make_pair(lhs_info_img, rhs_info_img),
                                           std::make_pair(lhs_info_buf, rhs_info_buf), n, k, b, DataType::F32);
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigReshapedBifrost::configure_G52_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);

    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;

    if (workload <= 323.4000f)
    {
        return configure_lhs_rhs_info(m, n, 2, 2, 8, 4, 8, false, false, false, true, false);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 8, 4, 2, 2, true, true, true, false, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigReshapedBifrost::configure_G76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    GEMMLHSMatrixInfo lhs_info_buf;
    GEMMRHSMatrixInfo rhs_info_buf;
    GEMMLHSMatrixInfo lhs_info_img;
    GEMMRHSMatrixInfo rhs_info_img;

    // Get lhs_info/rhs_info in case of OpenCL buffer
    if (n <= 4)
    {
        std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 2, 8, 16, 16, true, false, false, true);
    }
    else
    {
        std::tie(lhs_info_buf, rhs_info_buf) = configure_lhs_rhs_info(m, n, 4, 4, 2, 8, 16, false, false, false, true);
    }

    // Get lhs_info/rhs_info in case of OpenCL image
    // Condition on the GPU workload
    if ((m / 4) * (n / 4) >= 2560)
    {
        // Big workload
        std::tie(lhs_info_img, rhs_info_img) =
            configure_lhs_rhs_info(m, n, 4, 4, 4, 2, 8, true, true, true, false, true);
    }
    else
    {
        // Small workload
        std::tie(lhs_info_img, rhs_info_img) =
            configure_lhs_rhs_info(m, n, 2, 4, 4, 1, 1, true, true, true, false, true);
    }

    const TensorInfo  tensor_rhs_info(TensorShape(n, k, b), 1, DataType::F32);
    const TensorShape shape = compute_rhs_reshaped_shape(tensor_rhs_info, rhs_info_img);
    const TensorInfo  tensor_reshaped_info(shape, 1, DataType::F32);

    // In case of vector by matrix with few work-items, we use the OpenCL buffer rather than the OpenCL image2d
    const bool use_cl_image2d = (n <= 4) ? false : true;

    if (bool(validate_image2d_support_on_rhs(tensor_reshaped_info, rhs_info_img)) && use_cl_image2d)
    {
        return std::make_pair(lhs_info_img, rhs_info_img);
    }
    else
    {
        return std::make_pair(lhs_info_buf, rhs_info_buf);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigReshapedBifrost::configure_G76_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
    const float r_mk     = static_cast<float>(m) / static_cast<float>(k);

    if (workload <= 1595.2000f)
    {
        if (r_mk <= 2.1044f)
        {
            if (workload <= 870.4000f)
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
            return configure_lhs_rhs_info(m, n, 4, 2, 4, 2, 2, false, false, true, false, false);
        }
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 8, 4, 4, 2, true, true, true, false, false);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo>
ClGemmDefaultConfigReshapedBifrost::configure_G76_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    if (n <= 4)
    {
        return configure_lhs_rhs_info(m, n, 4, 2, 16, 4, 1, false, false, false, true);
    }
    else
    {
        return configure_lhs_rhs_info(m, n, 4, 4, 16, 2, 2, false, true, false, true);
    }
}
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
