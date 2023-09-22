/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_MAXUNPOOL_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_MAXUNPOOL_GENERIC_NEON_IMPL_H
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/wrapper/wrapper.h"
namespace arm_compute
{
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
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_MAXUNPOOL_GENERIC_NEON_IMPL_H
