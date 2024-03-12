/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_MATMULATTRIBUTES_H
#define ACL_ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_MATMULATTRIBUTES_H

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Attributes are backend-agnostic parameters (in addition to the input/output tensors) of an operator.
 */

/** MatMul attributes */
class MatMulAttributes
{
public:
    /* Get adjoint LHS flag value (transpose LHS before multiplication) */
    bool adj_lhs() const;

    /* Get adjoint RHS flag value (transpose RHS before multiplication) */
    bool adj_rhs() const;

    /* Set adjoint LHS flag value (transpose LHS before multiplication) */
    MatMulAttributes adj_lhs(bool adj_lhs);

    /* Set adjoint RHS flag value (transpose RHS before multiplication) */
    MatMulAttributes adj_rhs(bool adj_rhs);

private:
    bool _adj_lhs{false};
    bool _adj_rhs{false};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // ACL_ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_MATMULATTRIBUTES_H
