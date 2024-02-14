/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#include "CkwHelper.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
void get_coordinate_from_gws(GpuCkwScopedKernelWriter writer,
                             ckw::TileOperand        &coord,
                             const ckw::TileOperand  &gid,
                             ckw::TileOperand        &step)
{
    writer->op_binary(coord, ckw::BinaryOp::Mul, gid, step);
}

void get_coordinate_from_gws_overlapping_min(GpuCkwScopedKernelWriter writer,
                                             ckw::TileOperand        &coord,
                                             const ckw::TileOperand  &gid,
                                             ckw::TileOperand        &step,
                                             ckw::TileOperand        &shift_back,
                                             ckw::TileOperand        &const_0)
{
    // Applied formula: max((gid * step) - shift_back, 0)
    // where the shift_back operand is: (step - leftover_step) % step

    writer->op_binary(coord, ckw::BinaryOp::Mul, gid, step);
    writer->op_binary(coord, ckw::BinaryOp::Sub, coord, shift_back);
    writer->op_binary(coord, ckw::BinaryOp::Max, coord, const_0);
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
