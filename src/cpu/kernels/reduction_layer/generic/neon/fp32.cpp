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

#include "src/cpu/kernels/reduction_layer/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void reduce_RedOpYZW_complex_reduceZ_float32_4_2_SUM(const Window            &window,
                                                     const ITensor           *input,
                                                     ITensor                 *output,
                                                     const ReductionOperation op)
{
    Reducer<RedOpYZW_complex<float, 4, 2, ReductionOperation::SUM>>::reduceZ(
        window, input, output, RedOpYZW_complex<float, 4, 2, ReductionOperation::SUM>(), op);
}

void reduce_RedOpX_reduceX_float32_4(const Window            &window,
                                     const ITensor           *input,
                                     ITensor                 *output,
                                     const ReductionOperation op)
{
    return Reducer<RedOpX<float, 4>>::reduceX(window, input, output, RedOpX<float, 4>(), op);
}

void reduce_RedOpYZW_reduceY_float32_4(const Window            &window,
                                       const ITensor           *input,
                                       ITensor                 *output,
                                       const ReductionOperation op)
{
    return Reducer<RedOpYZW<float, 4>>::reduceY(window, input, output, RedOpYZW<float, 4>(), op);
}

void reduce_RedOpYZW_reduceZ_float32_4(const Window            &window,
                                       const ITensor           *input,
                                       ITensor                 *output,
                                       const ReductionOperation op)
{
    return Reducer<RedOpYZW<float, 4>>::reduceZ(window, input, output, RedOpYZW<float, 4>(), op);
}

void reduce_RedOpYZW_reduceW_float32_4(const Window            &window,
                                       const ITensor           *input,
                                       ITensor                 *output,
                                       const ReductionOperation op)
{
    return Reducer<RedOpYZW<float, 4>>::reduceW(window, input, output, RedOpYZW<float, 4>(), op);
}

} // namespace cpu
} // namespace arm_compute
