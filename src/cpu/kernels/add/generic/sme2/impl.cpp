/*
 * Copyright (c) 2024 Arm Limited.
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

#include "src/cpu/kernels/add/generic/sme2/impl.h"

#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
namespace cpu
{

bool add_q8_sme2_fixedpoint_possible(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return add_sub_q8_sme2_fixedpoint_possible(src0, src1, dst);
}

bool add_sub_q8_sme2_fixedpoint_possible(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    const auto        &in0_shape = src0->tensor_shape();
    const auto        &in1_shape = src1->tensor_shape();
    const unsigned int dst_dims  = dst->num_dimensions();
    // Does not support broadcasting on x
    // Does not support dims > 4D output, unless input shapes are identical (therefore collapsible)
    if (in0_shape.x() == in1_shape.x() && (in0_shape == in1_shape || dst_dims <= 4))
    {
        return true;
    }
    return false;
}

} // namespace cpu
} // namespace arm_compute
