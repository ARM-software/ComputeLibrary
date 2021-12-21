/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#if defined(ARM_COMPUTE_ENABLE_SVE2)
#include "arm_compute/core/Helpers.h"
#include "src/cpu/kernels/softmax/generic/sve2/impl.h"

namespace arm_compute
{
namespace cpu
{
void sve2_qasymm8_signed_softmax(const ITensor *in, const ITensor *max, void *const tmp,
                                 ITensor *out, const float beta, bool is_log, const Window &window)
{
    return sve2_softmax_logits_1d_quantized<qasymm8_signed_t>(in, max, tmp, out, beta, is_log, window);
}
}
} // namespace arm_compute
#endif //defined(ARM_COMPUTE_ENABLE_SVE2)
