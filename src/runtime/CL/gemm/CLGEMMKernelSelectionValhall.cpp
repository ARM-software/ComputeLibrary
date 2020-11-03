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
#include "src/runtime/CL/gemm/CLGEMMKernelSelectionValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "src/core/CL/gemm/CLGEMMHelpers.h"

#include <map>
#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
CLGEMMKernelSelectionValhall::CLGEMMKernelSelectionValhall(GPUTarget gpu)
    : ICLGEMMKernelSelection(gpu)
{
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::select_kernel(const CLGEMMKernelSelectionParams &params)
{
    // _target could be used in the future to have a dedicated heuristic for each GPU IP
    ARM_COMPUTE_UNUSED(_target);

    using FunctionExecutorPtr = CLGEMMKernelType (CLGEMMKernelSelectionValhall::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);

    // Default configurations for Valhall architectures
    static std::map<DataType, FunctionExecutorPtr> gemm_default_configs =
    {
        { DataType::F32, &CLGEMMKernelSelectionValhall::default_f32 },
        { DataType::F16, &CLGEMMKernelSelectionValhall::default_f16 },
        { DataType::QASYMM8, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QSYMM8, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMKernelSelectionValhall::default_q8 }
    };

    // Mali-G77 configurations
    static std::map<DataType, FunctionExecutorPtr> gemm_g77_configs =
    {
        { DataType::F32, &CLGEMMKernelSelectionValhall::default_f32 },
        { DataType::F16, &CLGEMMKernelSelectionValhall::g77_f16 },
        { DataType::QASYMM8, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QSYMM8, &CLGEMMKernelSelectionValhall::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMKernelSelectionValhall::default_q8 }
    };

    const DataType data_type = params.data_type;

    switch(_target)
    {
        case GPUTarget::G77:
            if(gemm_g77_configs.find(data_type) != gemm_g77_configs.end())
            {
                return (this->*gemm_g77_configs[data_type])(params.m, params.n, params.k, params.b, params.is_rhs_constant);
            }
            ARM_COMPUTE_ERROR("Not supported data type");
        default:
            if(gemm_default_configs.find(data_type) != gemm_default_configs.end())
            {
                return (this->*gemm_default_configs[data_type])(params.m, params.n, params.k, params.b, params.is_rhs_constant);
            }
            ARM_COMPUTE_ERROR("Not supported data type");
    }
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::default_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    return is_rhs_constant ? CLGEMMKernelType::RESHAPED_ONLY_RHS : CLGEMMKernelType::NATIVE_V1;
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::default_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    return is_rhs_constant ? CLGEMMKernelType::RESHAPED_ONLY_RHS : CLGEMMKernelType::NATIVE_V1;
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::g77_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    if (!is_rhs_constant)
    {
        return CLGEMMKernelType::NATIVE_V1;
    }

    if (m == 1)
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
    }

    const float r_mn = static_cast<float>(m) / static_cast<float>(n);
    const float r_mk = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk = static_cast<float>(n) / static_cast<float>(k);
    const float workload = (static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(b)) / 20.0f;

    if(r_mk <= 0.6817956566810608)
    {
        if(workload <= 801.6000061035156)
        {
            return CLGEMMKernelType::RESHAPED_ONLY_RHS;
        }
        else
        {
            if(r_mn <= 0.0839829258620739)
            {
                return CLGEMMKernelType::RESHAPED_ONLY_RHS;
            }
            else
            {
                if(r_mk <= 0.24917218834161758)
                {
                    return CLGEMMKernelType::RESHAPED;
                }
                else
                {
                    if(workload <= 2551.75)
                    {
                        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
                    }
                    else
                    {
                        if(workload <= 5061.574951171875)
                        {
                            return CLGEMMKernelType::RESHAPED_ONLY_RHS;
                        }
                        else
                        {
                            return CLGEMMKernelType::RESHAPED;
                        }
                    }
                }
            }
        }
    }
    else
    {
        if(r_mk <= 4.849947690963745)
        {
            if(workload <= 17618.4501953125)
            {
                if(workload <= 5224.699951171875)
                {
                    return CLGEMMKernelType::RESHAPED_ONLY_RHS;
                }
                else
                {
                    if(r_nk <= 0.7933054566383362)
                    {
                        return CLGEMMKernelType::RESHAPED;
                    }
                    else
                    {
                        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
                    }
                }
            }
            else
            {
                if(workload <= 20275.2001953125)
                {
                    return CLGEMMKernelType::RESHAPED;
                }
                else
                {
                    if(r_mk <= 3.07421875)
                    {
                        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
                    }
                    else
                    {
                        return CLGEMMKernelType::RESHAPED;
                    }
                }
            }
        }
        else
        {
            return CLGEMMKernelType::RESHAPED_ONLY_RHS;
        }
    }
}

CLGEMMKernelType CLGEMMKernelSelectionValhall::default_q8(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    if(is_rhs_constant)
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
    }
    else
    {
        return CLGEMMKernelType::NATIVE;
    }
}
} // namespace cl_gemm
} // namespace arm_compute
