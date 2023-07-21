/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_FUNCTION_INFO_MATMULINFO
#define ACL_ARM_COMPUTE_FUNCTION_INFO_MATMULINFO

namespace arm_compute
{
/** Class for holding information related to matrix multiplication function
 */
class MatMulInfo
{
public:
    /* Get Adjoint LHS flag value */
    bool adj_lhs() const
    {
        return _adj_lhs;
    }
    /* Get Adjoint RHS flag value */
    bool adj_rhs() const
    {
        return _adj_rhs;
    }
    /* Set Adjoint LHS flag */
    MatMulInfo &adj_lhs(bool adj_lhs)
    {
        _adj_lhs = adj_lhs;
        return *this;
    }
    /* Set Adjoint RHS flag */
    MatMulInfo &adj_rhs(bool adj_rhs)
    {
        _adj_rhs = adj_rhs;
        return *this;
    }

private:
    bool _adj_lhs{ false };
    bool _adj_rhs{ false };
};
} // namespace arm_compute
#endif /* ACL_ARM_COMPUTE_FUNCTION_INFO_MATMULINFO */
