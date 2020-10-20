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
#include "src/core/CL/gemm/native/CLGEMMNativeKernelConfigurationMidgard.h"

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
CLGEMMNativeKernelConfigurationMidgard::CLGEMMNativeKernelConfigurationMidgard(GPUTarget gpu)
    : ICLGEMMKernelConfiguration(gpu)
{
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationMidgard::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    using ConfigurationFunctionExecutorPtr = std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> (CLGEMMNativeKernelConfigurationMidgard::*)(unsigned int m, unsigned int n, unsigned int k,
                                             unsigned int b);

    // Configurations for Midgard architectures
    static std::map<DataType, ConfigurationFunctionExecutorPtr> default_configs =
    {
        { DataType::QASYMM8, &CLGEMMNativeKernelConfigurationMidgard::default_q8 },
        { DataType::QASYMM8_SIGNED, &CLGEMMNativeKernelConfigurationMidgard::default_q8 },
        { DataType::QSYMM8, &CLGEMMNativeKernelConfigurationMidgard::default_q8 },
        { DataType::QSYMM8_PER_CHANNEL, &CLGEMMNativeKernelConfigurationMidgard::default_q8 }
    };

    if(default_configs.find(data_type) != default_configs.end())
    {
        return (this->*default_configs[data_type])(m, n, k, b);
    }
    ARM_COMPUTE_ERROR("Not supported data type");
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMNativeKernelConfigurationMidgard::default_q8(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    const unsigned int m0 = std::min(m, static_cast<unsigned int>(4));
    const unsigned int n0 = std::min(n, static_cast<unsigned int>(4));

    return configure_lhs_rhs_info(m, n, m0, n0, 2, 1, 1, false, false, false, false);
}
} // namespace cl_gemm
} // namespace arm_compute