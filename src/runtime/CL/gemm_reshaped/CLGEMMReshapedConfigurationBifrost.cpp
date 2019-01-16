/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/CL/gemm_reshaped/CLGEMMReshapedConfigurationBifrost.h"

#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedConfigurationBifrost::configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    ARM_COMPUTE_ERROR_ON(data_type != DataType::F32);
    ARM_COMPUTE_UNUSED(data_type);

    const GPUTarget gpu_target = CLScheduler::get().target();
    switch(gpu_target)
    {
        case GPUTarget::G76:
            return configure_G76_f32(m, n, k, b);
        default:
            return configure_G7x_f32(m, n, k, b);
    }
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedConfigurationBifrost::configure_G7x_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    GEMMLHSMatrixInfo lhs_info;
    GEMMRHSMatrixInfo rhs_info;

    if(n <= 4)
    {
        // Configure GEMMLHSMatrixInfo
        lhs_info.m0         = 4;
        lhs_info.k0         = 8;
        lhs_info.v0         = lhs_info.m0 * 16 < m ? 2 : 16;
        lhs_info.interleave = true;
        lhs_info.transpose  = false;

        // Configure GEMMRHSMatrixInfo
        rhs_info.n0         = 2;
        rhs_info.k0         = lhs_info.k0;
        rhs_info.h0         = rhs_info.n0 * 16 < n ? 2 : 16;
        rhs_info.interleave = false;
        rhs_info.transpose  = true;
    }
    else
    {
        // Configure GEMMLHSMatrixInfo
        lhs_info.m0         = 5;
        lhs_info.k0         = 4;
        lhs_info.v0         = lhs_info.m0 * 2 < m ? 1 : 2;
        lhs_info.interleave = false;
        lhs_info.transpose  = false;

        // Configure GEMMRHSMatrixInfo
        rhs_info.n0         = 4;
        rhs_info.k0         = lhs_info.k0;
        rhs_info.h0         = rhs_info.n0 * 16 < n ? 2 : 16;
        rhs_info.interleave = true;
        rhs_info.transpose  = true;
    }

    return std::make_pair(lhs_info, rhs_info);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> CLGEMMReshapedConfigurationBifrost::configure_G76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_UNUSED(b);

    GEMMLHSMatrixInfo lhs_info;
    GEMMRHSMatrixInfo rhs_info;

    if(n <= 4)
    {
        // Configure GEMMLHSMatrixInfo
        lhs_info.m0         = 4;
        lhs_info.k0         = 8;
        lhs_info.v0         = lhs_info.m0 * 16 < m ? 2 : 16;
        lhs_info.interleave = true;
        lhs_info.transpose  = false;

        // Configure GEMMRHSMatrixInfo
        rhs_info.n0         = 2;
        rhs_info.k0         = lhs_info.k0;
        rhs_info.h0         = rhs_info.n0 * 16 < n ? 2 : 16;
        rhs_info.interleave = false;
        rhs_info.transpose  = true;
    }
    else
    {
        // Configure GEMMLHSMatrixInfo
        lhs_info.m0         = 4;
        lhs_info.k0         = 2;
        lhs_info.v0         = lhs_info.m0 * 8 < m ? 2 : 8;
        lhs_info.interleave = false;
        lhs_info.transpose  = false;

        // Configure GEMMRHSMatrixInfo
        rhs_info.n0         = 4;
        rhs_info.k0         = lhs_info.k0;
        rhs_info.h0         = rhs_info.n0 * 16 < n ? 2 : 16;
        rhs_info.interleave = false;
        rhs_info.transpose  = true;
    }

    return std::make_pair(lhs_info, rhs_info);
}
} // namespace cl_gemm
} // namespace arm_compute