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
#include "arm_compute/core/CL/gemm/CLGEMMHelpers.h"

#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_lhs_rhs_info(unsigned int m, unsigned int n, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
                                                                       bool lhs_interleave, bool rhs_interleave, bool lhs_transpose, bool rhs_transpose)
{
    GEMMLHSMatrixInfo lhs_info;
    GEMMRHSMatrixInfo rhs_info;

    // Configure GEMMLHSMatrixInfo
    lhs_info.m0         = m0;
    lhs_info.k0         = k0;
    lhs_info.v0         = ((m / (lhs_info.m0 * v0)) == 0) ? 1 : v0;
    lhs_info.interleave = lhs_interleave;
    lhs_info.transpose  = lhs_transpose;

    // Configure GEMMRHSMatrixInfo
    rhs_info.n0         = n0;
    rhs_info.k0         = lhs_info.k0;
    rhs_info.h0         = ((n / (rhs_info.n0 * h0)) == 0) ? 1 : h0;
    rhs_info.interleave = rhs_interleave;
    rhs_info.transpose  = rhs_transpose;

    return std::make_pair(lhs_info, rhs_info);
}
} // namespace cl_gemm
} // namespace arm_compute