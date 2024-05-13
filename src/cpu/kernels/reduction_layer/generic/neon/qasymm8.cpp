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
void reduce_RedOpX_reduceX_qasymm8(const Window            &window,
                                   const ITensor           *input,
                                   ITensor                 *output,
                                   const ReductionOperation op)
{
    return Reducer<RedOpX_quantized<uint8_t>>::reduceX(window, input, output, RedOpX_quantized<uint8_t>(), op);
}

void reduce_RedOpYZW_reduceY_qasymm8(const Window            &window,
                                     const ITensor           *input,
                                     ITensor                 *output,
                                     const ReductionOperation op)
{
    return Reducer<RedOpYZW_quantized<uint8_t>>::reduceY(window, input, output, RedOpYZW_quantized<uint8_t>(), op);
}

void reduce_RedOpYZW_reduceZ_qasymm8(const Window            &window,
                                     const ITensor           *input,
                                     ITensor                 *output,
                                     const ReductionOperation op)
{
    return Reducer<RedOpYZW_quantized<uint8_t>>::reduceZ(window, input, output, RedOpYZW_quantized<uint8_t>(), op);
}

void reduce_RedOpYZW_reduceW_qasymm8(const Window            &window,
                                     const ITensor           *input,
                                     ITensor                 *output,
                                     const ReductionOperation op)
{
    return Reducer<RedOpYZW_quantized<uint8_t>>::reduceW(window, input, output, RedOpYZW_quantized<uint8_t>(), op);
}
} // namespace cpu
} // namespace arm_compute
