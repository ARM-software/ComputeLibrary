/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#include "src/runtime/CL/gemm/CLGEMMDefaultTypeValhall.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"

#include <map>
#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
CLGEMMDefaultTypeValhall::CLGEMMDefaultTypeValhall(GPUTarget gpu)
    : ICLGEMMKernelSelection(gpu)
{
}

CLGEMMKernelType CLGEMMDefaultTypeValhall::select_kernel(const CLGEMMKernelSelectionParams &params)
{
    // _target could be used in the future to have a dedicated heuristic for each GPU IP
    ARM_COMPUTE_UNUSED(_target);

    using FunctionExecutorPtr = CLGEMMKernelType (CLGEMMDefaultTypeValhall::*)(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);

    // Default configurations for Valhall architectures
    static std::map<DataType, FunctionExecutorPtr> gemm_default_configs =
    {
        { DataType::F32, &CLGEMMDefaultTypeValhall::default_f32 },
        { DataType::F16, &CLGEMMDefaultTypeValhall::default_f16 },
        { DataType::QASYMM8, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QSYMM8, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMDefaultTypeValhall::default_q8 }
    };

    // Mali-G77 configurations
    static std::map<DataType, FunctionExecutorPtr> gemm_g77_configs =
    {
        { DataType::F32, &CLGEMMDefaultTypeValhall::default_f32 },
        { DataType::F16, &CLGEMMDefaultTypeValhall::g77_f16 },
        { DataType::QASYMM8, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QSYMM8, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMDefaultTypeValhall::default_q8 }
    };

    // Mali-G78 configurations
    static std::map<DataType, FunctionExecutorPtr> gemm_g78_configs =
    {
        { DataType::F32, &CLGEMMDefaultTypeValhall::g78_f32 },
        { DataType::F16, &CLGEMMDefaultTypeValhall::g78_f16 },
        { DataType::QASYMM8, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QSYMM8, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMDefaultTypeValhall::default_q8 }
    };

    // Mali-G715 and Mali-G615 configurations
    static std::map<DataType, FunctionExecutorPtr> gemm_g715_configs =
    {
        { DataType::F32, &CLGEMMDefaultTypeValhall::g715_f32 },
        { DataType::F16, &CLGEMMDefaultTypeValhall::g715_f16 },
        { DataType::QASYMM8, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QSYMM8, &CLGEMMDefaultTypeValhall::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMDefaultTypeValhall::default_q8 }
    };

    const DataType data_type = params.data_type;

    switch(_target)
    {
        case GPUTarget::G715:
        case GPUTarget::G615:
            if(gemm_g715_configs.find(data_type) != gemm_g715_configs.end())
            {
                return (this->*gemm_g715_configs[data_type])(params.m, params.n, params.k, params.b, params.is_rhs_constant);
            }
            ARM_COMPUTE_ERROR("Not supported data type");
        case GPUTarget::G78:
            if(gemm_g78_configs.find(data_type) != gemm_g78_configs.end())
            {
                return (this->*gemm_g78_configs[data_type])(params.m, params.n, params.k, params.b, params.is_rhs_constant);
            }
            ARM_COMPUTE_ERROR("Not supported data type");
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

CLGEMMKernelType CLGEMMDefaultTypeValhall::default_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    return is_rhs_constant ? CLGEMMKernelType::RESHAPED_ONLY_RHS : CLGEMMKernelType::NATIVE;
}

CLGEMMKernelType CLGEMMDefaultTypeValhall::default_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    return is_rhs_constant ? CLGEMMKernelType::RESHAPED_ONLY_RHS : CLGEMMKernelType::NATIVE;
}

CLGEMMKernelType CLGEMMDefaultTypeValhall::g77_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    if(!is_rhs_constant)
    {
        return CLGEMMKernelType::NATIVE;
    }

    if(m == 1)
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
    }

    const float r_mn     = static_cast<float>(m) / static_cast<float>(n);
    const float r_mk     = static_cast<float>(m) / static_cast<float>(k);
    const float r_nk     = static_cast<float>(n) / static_cast<float>(k);
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

CLGEMMKernelType CLGEMMDefaultTypeValhall::default_q8(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
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

CLGEMMKernelType CLGEMMDefaultTypeValhall::g78_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(b);

    if(!is_rhs_constant)
    {
        return CLGEMMKernelType::NATIVE;
    }

    if(m == 1)
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
    }

    if(n <= 272.0000f)
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
    }
    else
    {
        if(k <= 471.0000f)
        {
            return CLGEMMKernelType::RESHAPED_ONLY_RHS;
        }
        else
        {
            if(m <= 72.5000f)
            {
                return CLGEMMKernelType::RESHAPED_ONLY_RHS;
            }
            else
            {
                if(m <= 90.5000f)
                {
                    return CLGEMMKernelType::RESHAPED;
                }
                else
                {
                    if(k <= 2448.0000f)
                    {
                        if(n <= 756.0000f)
                        {
                            return CLGEMMKernelType::RESHAPED_ONLY_RHS;
                        }
                        else
                        {
                            return CLGEMMKernelType::RESHAPED;
                        }
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

CLGEMMKernelType CLGEMMDefaultTypeValhall::g78_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k, b);

    if(!is_rhs_constant)
    {
        return CLGEMMKernelType::NATIVE;
    }

    return CLGEMMKernelType::RESHAPED_ONLY_RHS;
}

CLGEMMKernelType CLGEMMDefaultTypeValhall::g715_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    if(!is_rhs_constant)
    {
        return default_f32(m, n, k, b, is_rhs_constant);
    }

    unsigned int best_m0;
    unsigned int best_n0;

    if(opencl::kernels::gemm::is_mmul_kernel_preferred(m, n, k, b, DataType::F32, best_m0, best_n0))
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS_MMUL;
    }
    else
    {
        return default_f32(m, n, k, b, is_rhs_constant);
    }
}

CLGEMMKernelType CLGEMMDefaultTypeValhall::g715_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant)
{
    if(!is_rhs_constant)
    {
        return g78_f16(m, n, k, b, is_rhs_constant);
    }

    unsigned int best_m0;
    unsigned int best_n0;

    if(opencl::kernels::gemm::is_mmul_kernel_preferred(m, n, k, b, DataType::F16, best_m0, best_n0))
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS_MMUL;
    }
    else
    {
        return g78_f16(m, n, k, b, is_rhs_constant);
    }
}

} // namespace cl_gemm
} // namespace arm_compute
