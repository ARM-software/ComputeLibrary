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
#ifndef __ARM_COMPUTE_ICLGEMMRESHAPEDCONFIGURATION_H__
#define __ARM_COMPUTE_ICLGEMMRESHAPEDCONFIGURATION_H__

#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** Basic interface for the GEMM selection */
class ICLGEMMReshapedConfiguration
{
public:
    /** Virtual destructor */
    virtual ~ICLGEMMReshapedConfiguration() = default;
    /** Given M, N, K and B, this method returns the @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo to be used with @ref CLGEMMMatrixMultiplyReshapedKernel
     *
     * @param[in] m         Number of rows LHS matrix
     * @param[in] n         Number of columns RHS matrix
     * @param[in] k         Number of columns LHS matrix or number of rows RHS matrix
     * @param[in] b         Batch size
     * @param[in] data_type Data type
     */
    virtual std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type) = 0;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_ICLGEMMRESHAPEDCONFIGURATION_H__ */
