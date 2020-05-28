/*
 * Copyright (c) 2020 ARM Limited.
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
#include "arm_compute/runtime/CL/gemm/CLGEMMKernelSelectionBifrost.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/gemm/CLGEMMHelpers.h"

#include <map>
#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
CLGEMMKernelSelectionBifrost::CLGEMMKernelSelectionBifrost(GPUTarget gpu)
    : ICLGEMMKernelSelection(gpu)
{
}

CLGEMMKernelType CLGEMMKernelSelectionBifrost::select_kernel(const CLGEMMKernelSelectionParams &params)
{
    // _target could be used in the future to have a dedicated heuristic for each GPU IP
    ARM_COMPUTE_UNUSED(_target);

    using FunctionExecutorPtr = CLGEMMKernelType (CLGEMMKernelSelectionBifrost::*)(unsigned int m, unsigned int n, unsigned int k, bool is_rhs_constant);

    // Default configurations for Bifrost architectures
    static std::map<DataType, FunctionExecutorPtr> gemm_default_configs =
    {
        { DataType::F32, &CLGEMMKernelSelectionBifrost::default_f32 },
        { DataType::F16, &CLGEMMKernelSelectionBifrost::default_f16 },
        { DataType::QASYMM8, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QSYMM8, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMKernelSelectionBifrost::default_q8 }
    };

    // Mali-G71 configurations
    static std::map<DataType, FunctionExecutorPtr> gemm_g71_configs =
    {
        { DataType::F32, &CLGEMMKernelSelectionBifrost::default_f32 },
        { DataType::F16, &CLGEMMKernelSelectionBifrost::g71_f16 },
        { DataType::QASYMM8, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QSYMM8, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMKernelSelectionBifrost::default_q8 }
    };

    // Mali-G76 configurations
    static std::map<DataType, FunctionExecutorPtr> gemm_g76_configs =
    {
        { DataType::F32, &CLGEMMKernelSelectionBifrost::g76_f32 },
        { DataType::F16, &CLGEMMKernelSelectionBifrost::default_f16 },
        { DataType::QASYMM8, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QSYMM8, &CLGEMMKernelSelectionBifrost::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMKernelSelectionBifrost::default_q8 }
    };

    const DataType data_type = params.data_type;

    switch(_target)
    {
        case GPUTarget::G71:
            if(gemm_g71_configs.find(data_type) != gemm_g71_configs.end())
            {
                return (this->*gemm_g71_configs[data_type])(params.m, params.n, params.k, params.is_rhs_constant);
            }
            ARM_COMPUTE_ERROR("Not supported data type");
        case GPUTarget::G76:
            if(gemm_g76_configs.find(data_type) != gemm_g76_configs.end())
            {
                return (this->*gemm_g76_configs[data_type])(params.m, params.n, params.k, params.is_rhs_constant);
            }
            ARM_COMPUTE_ERROR("Not supported data type");
        default:
            if(gemm_default_configs.find(data_type) != gemm_default_configs.end())
            {
                return (this->*gemm_default_configs[data_type])(params.m, params.n, params.k, params.is_rhs_constant);
            }
            ARM_COMPUTE_ERROR("Not supported data type");
    }
}

CLGEMMKernelType CLGEMMKernelSelectionBifrost::default_f32(unsigned int m, unsigned int n, unsigned int k, bool is_rhs_constant)
{
    CLGEMMKernelType gemm_type = CLGEMMKernelType::NATIVE_V1;

    if(is_rhs_constant)
    {
        if((m > 1) && (n < 16))
        {
            gemm_type = CLGEMMKernelType::RESHAPED_V1;
        }
        else if(m == 1)
        {
            gemm_type = CLGEMMKernelType::RESHAPED_ONLY_RHS;
        }
        else
        {
            if((k > 256) && (m > 4))
            {
                constexpr float alpha = 3.2f;
                constexpr float fact0 = 1.51f;
                constexpr float fact1 = 1.66f;
                constexpr float ops   = 12.0f;
                const float     scale = k > 1024 ? 1.07f : 1.0f;
                gemm_type             = (alpha + ((n * fact0) / ops) < ((fact1 * n * scale) / ops)) ? CLGEMMKernelType::RESHAPED_V1 : CLGEMMKernelType::NATIVE_V1;
            }
            else
            {
                gemm_type = CLGEMMKernelType::NATIVE_V1;
            }
        }

        const auto workload = static_cast<float>((m * n) / 20.0f);

        gemm_type = ((workload > 1600.0f) && (gemm_type == CLGEMMKernelType::RESHAPED_V1)) ? CLGEMMKernelType::RESHAPED : gemm_type;
    }

    return gemm_type;
}

CLGEMMKernelType CLGEMMKernelSelectionBifrost::default_f16(unsigned int m, unsigned int n, unsigned int k, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(n, k);
    if(is_rhs_constant)
    {
        if(m == 1)
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
        return CLGEMMKernelType::NATIVE_V1;
    }
}

CLGEMMKernelType CLGEMMKernelSelectionBifrost::default_q8(unsigned int m, unsigned int n, unsigned int k, bool is_rhs_constant)
{
    ARM_COMPUTE_UNUSED(m, n, k);

    if(is_rhs_constant)
    {
        return CLGEMMKernelType::RESHAPED_ONLY_RHS;
    }
    else
    {
        return CLGEMMKernelType::NATIVE;
    }
}

CLGEMMKernelType CLGEMMKernelSelectionBifrost::g76_f32(unsigned int m, unsigned int n, unsigned int k, bool is_rhs_constant)
{
    CLGEMMKernelType gemm_type = CLGEMMKernelType::NATIVE_V1;

    if(is_rhs_constant)
    {
        if((m > 1) && (n < 16))
        {
            gemm_type = CLGEMMKernelType::RESHAPED;
        }
        else if(m == 1)
        {
            gemm_type = CLGEMMKernelType::RESHAPED_ONLY_RHS;
        }
        else
        {
            if((k > 256) && (m > 4))
            {
                gemm_type = CLGEMMKernelType::RESHAPED;
            }
            else
            {
                gemm_type = CLGEMMKernelType::RESHAPED_ONLY_RHS;
            }
        }
    }

    return gemm_type;
}

CLGEMMKernelType CLGEMMKernelSelectionBifrost::g71_f16(unsigned int m, unsigned int n, unsigned int k, bool is_rhs_constant)
{
    if(is_rhs_constant)
    {
        if(m == 1)
        {
            if(n > k)
            {
                return CLGEMMKernelType::NATIVE_V1;
            }
            else
            {
                return CLGEMMKernelType::RESHAPED_ONLY_RHS;
            }
        }
        else
        {
            return CLGEMMKernelType::RESHAPED;
        }
    }
    else
    {
        return CLGEMMKernelType::NATIVE_V1;
    }
}
} // namespace cl_gemm
} // namespace arm_compute
