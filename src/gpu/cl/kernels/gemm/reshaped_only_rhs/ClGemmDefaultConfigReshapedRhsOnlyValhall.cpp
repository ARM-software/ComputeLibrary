/*
 * Copyright (c) 2020-2023 Arm Limited.
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
#include "src/gpu/cl/kernels/gemm/reshaped_only_rhs/ClGemmDefaultConfigReshapedRhsOnlyValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
#include "src/runtime/CL/gemm/CLGEMMDefaultTypeValhall.h"

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

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G710(&ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_f32,
                                                                     &ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G710_f16,
                                                                     &ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_u8);

    CLGEMMConfigArray<ConfigurationFunctionExecutorPtr> configs_G715(&ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G715_f32,
                                                                     &ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G715_f16,
                                                                     &ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G77_u8);

    ConfigurationFunctionExecutorPtr func = nullptr;

    switch(_target)
    {
        case GPUTarget::G78:
            func = configs_G78.get_function(data_type);
            break;
        case GPUTarget::G710:
        case GPUTarget::G610:
            func = configs_G710.get_function(data_type);
            break;
        case GPUTarget::G715:
        case GPUTarget::G615:
            func = configs_G715.get_function(data_type);
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
    const GeMMConfigsMatrix configs_1nkb_best =
    {
        { 1, 8984, 640, 1, 1, 8, 8, 1, 0, 1, 1, 1, 1, 0 },
        { 1, 420, 392, 1, 1, 2, 8, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 644, 5288, 1, 1, 2, 8, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 6512, 6404, 1, 1, 4, 8, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 5304, 640, 1, 1, 4, 4, 1, 0, 1, 0, 1, 1, 0 },
        { 1, 1352, 1520, 1, 1, 2, 8, 1, 0, 1, 1, 1, 1, 0 },
        { 1, 4096, 25088, 1, 1, 2, 16, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 732, 8988, 1, 1, 2, 8, 1, 0, 1, 0, 1, 0, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_n_small_best =
    {
        { 102400, 4, 96, 1, 2, 2, 16, 1, 4, 1, 1, 1, 1, 0 },
        { 102400, 2, 96, 1, 1, 2, 16, 1, 0, 1, 0, 1, 1, 1 },
        { 16384, 4, 128, 1, 1, 2, 16, 1, 0, 1, 0, 1, 1, 1 },
        { 16384, 2, 128, 1, 1, 2, 16, 1, 0, 1, 1, 1, 1, 1 }
    };

    const GeMMConfigsMatrix configs_mnkb_n_small_fallback =
    {
        { 102400, 4, 96, 1, 2, 2, 16, 1, 4, 1, 1, 1, 1, 0 },
        { 102400, 2, 96, 1, 1, 2, 16, 1, 0, 1, 1, 1, 1, 0 },
        { 16384, 4, 128, 1, 2, 2, 16, 1, 2, 1, 1, 1, 1, 0 },
        { 16384, 2, 128, 1, 1, 2, 16, 1, 0, 1, 1, 1, 1, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_m_gt_n_best =
    {
        { 25584, 88, 16, 1, 4, 8, 4, 1, 8, 1, 1, 1, 0, 0 },
        { 25584, 16, 68, 1, 4, 4, 8, 1, 16, 1, 1, 1, 0, 1 },
        { 369664, 32, 28, 1, 5, 4, 4, 1, 64, 1, 1, 1, 0, 1 },
        { 65792, 44, 24, 1, 4, 8, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 23036, 56, 736, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 1 },
        { 90968, 40, 600, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 1 },
        { 8944, 32, 776, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 1 },
        { 50176, 64, 300, 1, 4, 8, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 16544, 104, 160, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 1 },
        { 12604, 60, 160, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 1 },
        { 29584, 32, 28, 1, 4, 4, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 12544, 32, 27, 1, 2, 8, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 2688, 136, 1492, 1, 8, 4, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 3728, 96, 196, 1, 4, 8, 4, 1, 128, 1, 1, 1, 0, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_m_gt_n_fallback =
    {
        { 25584, 88, 16, 1, 4, 8, 4, 1, 8, 1, 1, 1, 0, 0 },
        { 25584, 16, 68, 1, 2, 4, 8, 1, 4, 1, 1, 1, 0, 0 },
        { 369664, 32, 28, 1, 5, 4, 4, 1, 256, 1, 1, 1, 0, 0 },
        { 65792, 44, 24, 1, 4, 8, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 23036, 56, 736, 1, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 90968, 40, 600, 1, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 8944, 32, 776, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 0 },
        { 50176, 64, 300, 1, 4, 8, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 16544, 104, 160, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 0 },
        { 12604, 60, 160, 1, 4, 4, 8, 1, 256, 1, 1, 1, 0, 0 },
        { 29584, 32, 28, 1, 4, 4, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 12544, 32, 27, 1, 2, 8, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 2688, 136, 1492, 1, 8, 4, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 3728, 96, 196, 1, 4, 8, 4, 1, 128, 1, 1, 1, 0, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_n_gt_m_best =
    {
        { 24, 488, 88, 1, 2, 4, 16, 1, 4, 1, 1, 1, 0, 0 },
        { 49, 1024, 512, 1, 4, 4, 8, 1, 128, 1, 1, 1, 0, 1 },
        { 49, 1024, 1024, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 1 },
    };

    const GeMMConfigsMatrix configs_mnkb_n_gt_m_fallback =
    {
        { 24, 488, 88, 1, 2, 4, 16, 1, 4, 1, 1, 1, 0, 0 },
        { 49, 1024, 512, 1, 4, 4, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 49, 1024, 1024, 1, 4, 4, 8, 1, 256, 1, 1, 1, 0, 0 },
    };

    const GeMMConfigsMatrix configs_mnkb_squared_best =
    {
        { 72, 92, 136, 1, 2, 2, 8, 1, 128, 1, 1, 1, 1, 0 },
        { 268, 824, 5076, 1, 4, 8, 4, 1, 256, 1, 1, 1, 0, 0 },
        { 180, 420, 952, 1, 4, 4, 8, 1, 64, 1, 1, 1, 0, 1 },
        { 1000, 152, 304, 1, 4, 4, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 272, 400, 2116, 1, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 196, 512, 512, 1, 5, 4, 4, 1, 64, 1, 1, 1, 0, 1 },
        { 24, 88, 236, 1, 2, 2, 8, 1, 64, 1, 1, 1, 1, 0 },
        { 24, 88, 488, 1, 2, 2, 8, 1, 64, 1, 1, 1, 1, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_squared_fallback =
    {
        { 72, 92, 136, 1, 2, 2, 8, 1, 128, 1, 1, 1, 1, 0 },
        { 268, 824, 5076, 1, 4, 8, 4, 1, 256, 1, 1, 1, 0, 0 },
        { 180, 420, 952, 1, 4, 4, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 1000, 152, 304, 1, 4, 4, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 272, 400, 2116, 1, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 196, 512, 512, 1, 5, 4, 4, 1, 256, 1, 1, 1, 0, 0 },
        { 24, 88, 236, 1, 2, 2, 8, 1, 64, 1, 1, 1, 1, 0 },
        { 24, 88, 488, 1, 2, 2, 8, 1, 64, 1, 1, 1, 1, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_best_batched =
    {
        { 3136, 64, 64, 36, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 4096, 48, 32, 36, 4, 4, 8, 1, 64, 1, 1, 1, 0, 1 },
        { 688, 92, 68, 32, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 24, 464, 412, 24, 4, 4, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 112, 184, 144, 28, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 5776, 64, 32, 36, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 1568, 64, 40, 36, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 2920, 64, 64, 24, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_fallback_batched =
    {
        { 3136, 64, 64, 36, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 4096, 48, 32, 36, 4, 4, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 688, 92, 68, 32, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 24, 464, 412, 24, 4, 4, 8, 1, 128, 1, 1, 1, 0, 0 },
        { 112, 184, 144, 28, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 5776, 64, 32, 36, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 1568, 64, 40, 36, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 },
        { 2920, 64, 64, 24, 4, 8, 4, 1, 64, 1, 1, 1, 0, 0 }
    };

    const GeMMConfigsMatrix *configs_best_to_use     = nullptr;
    const GeMMConfigsMatrix *configs_fallback_to_use = nullptr;

    if(b == 1)
    {
        constexpr float        ratio_m_gt_n = 10.f;
        constexpr float        ratio_n_gt_m = 0.1f;
        constexpr unsigned int n_small_thr  = 4;
        const float            ratio        = static_cast<float>(m) / static_cast<float>(n);

        if(m == 1)
        {
            // We do not need fallback in this case, as we never use cl_image for the rhs tensor
            configs_best_to_use     = &configs_1nkb_best;
            configs_fallback_to_use = &configs_1nkb_best;
        }
        else if(n <= n_small_thr && ratio > ratio_m_gt_n)
        {
            configs_best_to_use     = &configs_mnkb_n_small_best;
            configs_fallback_to_use = &configs_mnkb_n_small_fallback;
        }
        else if(ratio > ratio_m_gt_n)
        {
            configs_best_to_use     = &configs_mnkb_m_gt_n_best;
            configs_fallback_to_use = &configs_mnkb_m_gt_n_fallback;
        }
        else if(ratio < ratio_n_gt_m)
        {
            configs_best_to_use     = &configs_mnkb_n_gt_m_best;
            configs_fallback_to_use = &configs_mnkb_n_gt_m_fallback;
        }
        else
        {
            configs_best_to_use     = &configs_mnkb_squared_best;
            configs_fallback_to_use = &configs_mnkb_squared_fallback;
        }
    }
    else
    {
        configs_best_to_use     = &configs_mnkb_best_batched;
        configs_fallback_to_use = &configs_mnkb_fallback_batched;
    }

    GEMMLHSMatrixInfo lhs_info0;
    GEMMRHSMatrixInfo rhs_info0;
    GEMMLHSMatrixInfo lhs_info1;
    GEMMRHSMatrixInfo rhs_info1;

    std::tie(lhs_info0, rhs_info0) = find_lhs_rhs_info(*configs_best_to_use, m, n, k, b);
    std::tie(lhs_info1, rhs_info1) = find_lhs_rhs_info(*configs_fallback_to_use, m, n, k, b);

    return select_lhs_rhs_info(std::make_pair(lhs_info0, rhs_info0),
                               std::make_pair(lhs_info1, rhs_info1),
                               n, k, b, DataType::F16);
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
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;
    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float r_mk     = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);

    if(m == 1)
    {
        const GeMMConfigsMatrix configs_mnkb_best =
        {
            { 1, 8984, 640, 1, 1, 4, 2, 1, 0, 1, 0, 1, 1, 0 },
            { 1, 420, 392, 1, 1, 2, 4, 1, 0, 1, 0, 1, 0, 0 },
            { 1, 644, 5288, 1, 1, 2, 4, 1, 0, 1, 0, 1, 0, 0 },
            { 1, 6512, 6404, 1, 1, 2, 2, 1, 0, 1, 0, 1, 1, 0 },
            { 1, 5304, 640, 1, 1, 2, 2, 1, 0, 1, 0, 1, 0, 0 },
            { 1, 1352, 1520, 1, 1, 2, 4, 1, 0, 1, 0, 1, 0, 0 },
            { 1, 4096, 25088, 1, 1, 2, 4, 1, 0, 1, 0, 1, 0, 0 },
            { 1, 732, 8988, 1, 1, 2, 4, 1, 0, 1, 0, 1, 0, 0 }
        };

        return find_lhs_rhs_info(configs_mnkb_best, m, n, k, b);
    }
    else
    {
        if(workload <= 1384.8000f)
        {
            if(r_nk <= 0.8333f)
            {
                if(r_mk <= 0.9119f)
                {
                    return configure_lhs_rhs_info(m, n, 2, 2, 16, 1, 4, 0, 1, 0, 1, 1);
                }
                else
                {
                    if(r_nk <= 0.1181f)
                    {
                        return configure_lhs_rhs_info(m, n, 2, 2, 8, 1, 32, 0, 0, 1, 0, 0);
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 32, 0, 1, 1, 0, 0);
                    }
                }
            }
            else
            {
                if(r_mk <= 1.0013f)
                {
                    return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 32, 0, 1, 1, 0, 1);
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 5, 4, 8, 1, 4, 0, 1, 1, 0, 1);
                }
            }
        }
        else
        {
            if(workload <= 11404.7998f)
            {
                if(r_mk <= 2.2884f)
                {
                    if(r_nk <= 0.9286f)
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 4, 0, 1, 1, 0, 1);
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 4, 4, 8, 1, 32, 0, 1, 1, 0, 1);
                    }
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 5, 4, 8, 1, 4, 0, 1, 1, 0, 1);
                }
            }
            else
            {
                if(r_nk <= 1.1926f)
                {
                    if(r_mn <= 1385.7917f)
                    {
                        return configure_lhs_rhs_info(m, n, 6, 4, 8, 1, 4, 0, 1, 1, 0, 1);
                    }
                    else
                    {
                        return configure_lhs_rhs_info(m, n, 2, 8, 8, 1, 32, 0, 1, 1, 0, 0);
                    }
                }
                else
                {
                    return configure_lhs_rhs_info(m, n, 6, 4, 8, 1, 32, 0, 1, 1, 0, 1);
                }
            }
        }
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G715_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    unsigned int best_m0;
    unsigned int best_n0;

    if(is_mmul_kernel_preferred(m, n, k, b, DataType::F32, best_m0, best_n0))
    {
        return configure_lhs_rhs_info(m, n, best_m0, best_n0, 1, 1, 4, false, true, false, false, true);
    }
    else
    {
        return configure_G77_f32(m, n, k, b);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G710_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    const GeMMConfigsMatrix configs_1nkb_best =
    {
        { 1, 8984, 640, 1, 1, 2, 2, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 420, 392, 1, 1, 2, 8, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 644, 5288, 1, 1, 2, 8, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 6512, 6404, 1, 1, 2, 4, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 5304, 640, 1, 1, 2, 4, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 1352, 1520, 1, 1, 2, 4, 1, 0, 1, 0, 1, 0, 0 },
        { 1, 4096, 25088, 1, 1, 2, 8, 1, 0, 1, 0, 1, 1, 0 },
        { 1, 732, 8988, 1, 1, 2, 8, 1, 0, 1, 0, 1, 0, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_n_small_best =
    {
        { 102400, 4, 96, 1, 1, 2, 16, 1, 0, 1, 0, 1, 0, 0 },
        { 102400, 2, 96, 1, 1, 2, 16, 1, 0, 1, 0, 1, 0, 0 },
        { 16384, 4, 128, 1, 1, 2, 16, 1, 0, 1, 0, 1, 0, 0 },
        { 16384, 2, 128, 1, 1, 2, 16, 1, 0, 1, 0, 1, 0, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_m_gt_n_best =
    {
        { 25584, 88, 16, 1, 4, 8, 4, 1, 4, 1, 1, 1, 0, 0 },
        { 25584, 16, 68, 1, 2, 4, 16, 1, 8, 1, 1, 1, 0, 1 },
        { 369664, 32, 28, 1, 2, 8, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 65792, 44, 24, 1, 4, 8, 4, 1, 8, 1, 1, 1, 0, 0 },
        { 23036, 56, 736, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 90968, 40, 600, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 8944, 32, 776, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 2688, 136, 1492, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 50176, 64, 300, 1, 4, 8, 4, 1, 8, 1, 1, 1, 0, 1 },
        { 16544, 104, 160, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 12604, 60, 160, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 3728, 96, 196, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 29584, 32, 28, 1, 2, 8, 4, 1, 16, 1, 1, 1, 0, 0 },
        { 12544, 32, 27, 1, 2, 8, 8, 1, 16, 1, 1, 1, 0, 0 },
    };

    const GeMMConfigsMatrix configs_mnkb_m_gt_n_fallback =
    {
        { 25584, 88, 16, 1, 4, 8, 4, 1, 4, 1, 1, 1, 0, 0 },
        { 25584, 16, 68, 1, 2, 4, 8, 1, 4, 1, 1, 1, 1, 0 },
        { 369664, 32, 28, 1, 2, 8, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 65792, 44, 24, 1, 4, 8, 4, 1, 8, 1, 1, 1, 0, 0 },
        { 23036, 56, 736, 1, 4, 8, 4, 1, 16, 1, 1, 1, 0, 0 },
        { 90968, 40, 600, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 0 },
        { 8944, 32, 776, 1, 2, 8, 8, 1, 16, 1, 1, 1, 0, 0 },
        { 2688, 136, 1492, 1, 4, 4, 8, 1, 8, 1, 1, 1, 0, 0 },
        { 50176, 64, 300, 1, 4, 8, 4, 1, 128, 1, 1, 1, 0, 0 },
        { 16544, 104, 160, 1, 4, 8, 4, 1, 16, 1, 1, 1, 0, 0 },
        { 12604, 60, 160, 1, 2, 8, 8, 1, 8, 1, 1, 1, 0, 0 },
        { 3728, 96, 196, 1, 2, 8, 8, 1, 64, 1, 1, 1, 0, 0 },
        { 29584, 32, 28, 1, 2, 8, 4, 1, 16, 1, 1, 1, 0, 0 },
        { 12544, 32, 27, 1, 2, 8, 8, 1, 16, 1, 1, 1, 0, 0 },
    };

    const GeMMConfigsMatrix configs_mnkb_n_gt_m_best =
    {
        { 24, 488, 88, 1, 2, 2, 8, 1, 8, 1, 1, 1, 1, 0 },
        { 49, 1024, 512, 1, 2, 4, 8, 1, 8, 1, 1, 1, 1, 0 },
        { 49, 1024, 1024, 1, 2, 4, 8, 1, 4, 1, 1, 1, 1, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_n_gt_m_fallback =
    {
        { 24, 488, 88, 1, 2, 2, 8, 1, 8, 1, 1, 1, 1, 0 },
        { 49, 1024, 512, 1, 2, 4, 8, 1, 8, 1, 1, 1, 1, 0 },
        { 49, 1024, 1024, 1, 2, 4, 8, 1, 4, 1, 1, 1, 1, 0 }
    };

    const GeMMConfigsMatrix configs_mnkb_squared_best =
    {
        { 24, 88, 236, 1, 2, 2, 8, 1, 4, 1, 1, 1, 1, 0 },
        { 24, 88, 488, 1, 2, 2, 8, 1, 4, 1, 1, 1, 1, 0 },
        { 72, 92, 136, 1, 2, 2, 8, 1, 32, 1, 1, 1, 1, 0 },
        { 268, 824, 5076, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 180, 420, 952, 1, 4, 4, 8, 1, 16, 1, 1, 1, 0, 1 },
        { 1000, 152, 304, 1, 4, 8, 4, 1, 32, 1, 1, 1, 0, 0 },
        { 272, 400, 2116, 1, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 196, 512, 512, 1, 5, 2, 8, 1, 4, 1, 1, 1, 1, 1 },
    };

    const GeMMConfigsMatrix configs_mnkb_squared_fallback =
    {
        { 24, 88, 236, 1, 2, 2, 8, 1, 4, 1, 1, 1, 1, 0 },
        { 24, 88, 488, 1, 2, 2, 8, 1, 4, 1, 1, 1, 1, 0 },
        { 72, 92, 136, 1, 2, 2, 8, 1, 32, 1, 1, 1, 1, 0 },
        { 268, 824, 5076, 1, 4, 8, 4, 1, 8, 1, 1, 1, 0, 0 },
        { 180, 420, 952, 1, 5, 2, 8, 1, 8, 1, 1, 1, 1, 0 },
        { 1000, 152, 304, 1, 4, 8, 4, 1, 32, 1, 1, 1, 0, 0 },
        { 272, 400, 2116, 1, 2, 8, 4, 1, 4, 1, 1, 1, 0, 0 },
        { 196, 512, 512, 1, 5, 2, 8, 1, 8, 1, 1, 1, 1, 0 },
    };

    const GeMMConfigsMatrix configs_mnkb_best_batched =
    {
        { 3136, 64, 64, 36, 4, 8, 4, 1, 16, 1, 1, 1, 0, 1 },
        { 4096, 48, 32, 36, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 688, 92, 68, 32, 4, 8, 4, 1, 32, 1, 1, 1, 0, 1 },
        { 24, 464, 412, 24, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 112, 184, 144, 28, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 5776, 64, 32, 36, 4, 4, 8, 1, 4, 1, 1, 1, 0, 1 },
        { 1568, 64, 40, 36, 4, 8, 4, 1, 8, 1, 1, 1, 0, 1 },
        { 2920, 64, 64, 24, 4, 8, 4, 1, 8, 1, 1, 1, 0, 1 }
    };

    const GeMMConfigsMatrix configs_mnkb_fallback_batched =
    {
        { 3136, 64, 64, 36, 4, 8, 4, 1, 8, 1, 1, 1, 0, 0 },
        { 4096, 48, 32, 36, 4, 4, 8, 1, 64, 1, 1, 1, 0, 0 },
        { 688, 92, 68, 32, 4, 8, 4, 1, 32, 1, 1, 1, 0, 0 },
        { 24, 464, 412, 24, 2, 8, 4, 1, 32, 1, 1, 1, 0, 0 },
        { 112, 184, 144, 28, 4, 4, 8, 1, 8, 1, 1, 1, 0, 0 },
        { 5776, 64, 32, 36, 2, 8, 8, 1, 32, 1, 1, 1, 0, 0 },
        { 1568, 64, 40, 36, 4, 8, 4, 1, 16, 1, 1, 1, 0, 0 },
        { 2920, 64, 64, 24, 4, 8, 4, 1, 8, 1, 1, 1, 0, 0 }
    };

    const GeMMConfigsMatrix *configs_best_to_use     = nullptr;
    const GeMMConfigsMatrix *configs_fallback_to_use = nullptr;

    if(b == 1)
    {
        constexpr float        ratio_m_gt_n = 10.f;
        constexpr float        ratio_n_gt_m = 0.1f;
        constexpr unsigned int n_small_thr  = 4;
        const float            ratio        = static_cast<float>(m) / static_cast<float>(n);

        if(m == 1)
        {
            // We do not need fallback in this case, as we never use cl_image for the rhs tensor
            configs_best_to_use     = &configs_1nkb_best;
            configs_fallback_to_use = &configs_1nkb_best;
        }
        else if(n <= n_small_thr && ratio > ratio_m_gt_n)
        {
            configs_best_to_use     = &configs_mnkb_n_small_best;
            configs_fallback_to_use = &configs_mnkb_n_small_best;
        }
        else if(ratio > ratio_m_gt_n)
        {
            configs_best_to_use     = &configs_mnkb_m_gt_n_best;
            configs_fallback_to_use = &configs_mnkb_m_gt_n_fallback;
        }
        else if(ratio < ratio_n_gt_m)
        {
            configs_best_to_use     = &configs_mnkb_n_gt_m_best;
            configs_fallback_to_use = &configs_mnkb_n_gt_m_fallback;
        }
        else
        {
            configs_best_to_use     = &configs_mnkb_squared_best;
            configs_fallback_to_use = &configs_mnkb_squared_fallback;
        }
    }
    else
    {
        configs_best_to_use     = &configs_mnkb_best_batched;
        configs_fallback_to_use = &configs_mnkb_fallback_batched;
    }

    GEMMLHSMatrixInfo lhs_info0;
    GEMMRHSMatrixInfo rhs_info0;
    GEMMLHSMatrixInfo lhs_info1;
    GEMMRHSMatrixInfo rhs_info1;

    std::tie(lhs_info0, rhs_info0) = find_lhs_rhs_info(*configs_best_to_use, m, n, k, b);
    std::tie(lhs_info1, rhs_info1) = find_lhs_rhs_info(*configs_fallback_to_use, m, n, k, b);

    return select_lhs_rhs_info(std::make_pair(lhs_info0, rhs_info0),
                               std::make_pair(lhs_info1, rhs_info1),
                               n, k, b, DataType::F16);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> ClGemmDefaultConfigReshapedRhsOnlyValhall::configure_G715_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    unsigned int best_m0;
    unsigned int best_n0;

    if(is_mmul_kernel_preferred(m, n, k, b, DataType::F16, best_m0, best_n0))
    {
        return configure_lhs_rhs_info(m, n, best_m0, best_n0, 1, 1, 4, false, true, false, false, true);
    }
    else
    {
        return configure_G78_f16(m, n, k, b);
    }
}
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
