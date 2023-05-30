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
#ifndef ARM_COMPUTE_MATMULINFO_H
#define ARM_COMPUTE_MATMULINFO_H

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Size3D.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/core/utils/misc/Macros.h"
#include "support/Bfloat16.h"
#include "support/Half.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

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
    /* Get Fused Activation Layer Info */
    ActivationLayerInfo fused_activation() const
    {
        return _fused_act;
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
    /* Set Fused Activation Layer Info */
    MatMulInfo &fused_activation(const ActivationLayerInfo &act_info)
    {
        _fused_act = act_info;
        return *this;
    }

private:
    bool                _adj_lhs{ false };
    bool                _adj_rhs{ false };
    ActivationLayerInfo _fused_act{}; // disabled by default
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_MATMULINFO_H */
