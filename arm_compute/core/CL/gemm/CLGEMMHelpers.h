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
#ifndef __ARM_COMPUTE_CLGEMMHELPERS_H__
#define __ARM_COMPUTE_CLGEMMHELPERS_H__

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace cl_gemm
{
/** Configure @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo
 *
 * @param[in] m              Number of rows (M) in the LHS matrix not reshaped
 * @param[in] n              Number of columns (N) in the RHS matrix not reshaped
 * @param[in] m0             Number of rows processed by each thread/work-item
 * @param[in] n0             Number of columns processed by each thread/work-item
 * @param[in] k0             Number of inner accumulation performed by each thread/work-item
 * @param[in] v0             Number of vertical blocks of size (m0xk0) stored on the same output row
 * @param[in] h0             Number of horizontal blocks of size (k0xn0) stored on the same output row
 * @param[in] lhs_interleave True if the v0 (m0xk0) blocks have to be interleaved in the output row
 * @param[in] rhs_interleave True if the h0 (k0xn0) blocks have to be interleaved in the output row
 * @param[in] lhs_transpose  True if the (m0xk0) block has to be transposed before been stored
 * @param[in] rhs_transpose  True if the (k0xn0) block has to be transposed before been stored
 *
 * @return @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo
 */
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_lhs_rhs_info(unsigned int m, unsigned int n, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
                                                                       bool lhs_interleave, bool rhs_interleave, bool lhs_transpose, bool rhs_transpose);
} // namespace cl_gemm
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGEMMHELPERS_H__ */
