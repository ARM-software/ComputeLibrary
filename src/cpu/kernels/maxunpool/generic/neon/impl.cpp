/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#include "src/cpu/kernels/maxunpool/generic/neon/impl.h"
namespace arm_compute
{
class ITensor;
class Window;
namespace cpu
{
template <typename T>
void max_unpooling(const ITensor *input, const ITensor *indices, ITensor *output, const Window &window)
{
    Iterator  input_itr(input, window);
    Iterator  indices_itr(indices, window);
    auto      out_ptr      = reinterpret_cast<T *>(output->buffer());
    const int out_stride_w = static_cast<int>(output->info()->strides_in_bytes()[3]);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto vindices                                         = reinterpret_cast<uint32_t *>(indices_itr.ptr());
        auto vinput                                           = reinterpret_cast<T *>(input_itr.ptr());
        out_ptr[id[3] * out_stride_w / sizeof(T) + *vindices] = *vinput;
    },
    input_itr, indices_itr);
}
template void max_unpooling<float>(const ITensor *input, const ITensor *indices, ITensor *output, const Window &window);
template void max_unpooling<int8_t>(const ITensor *input, const ITensor *indices, ITensor *output, const Window &window);
template void max_unpooling<uint8_t>(const ITensor *input, const ITensor *indices, ITensor *output, const Window &window);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
template void max_unpooling<float16_t>(const ITensor *input, const ITensor *indices, ITensor *output, const Window &window);
#endif //defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
} // namespace cpu
} // namespace arm_compute
