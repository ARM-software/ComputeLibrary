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
#include "arm_compute/function_info/ScatterInfo.h"

#include "src/cpu/kernels/scatter/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void scatter_fp32_neon(const ITensor     *src,
                       const ITensor     *indices,
                       ITensor           *dst,
                       const ScatterInfo &scatter_info,
                       const Window      &window,
                       const int          data_block_length)
{
    switch (scatter_info.func)
    {
        case ScatterFunction::Update:
            scatter_neon_fp32<ScatterFunction::Update>(src, indices, dst, window, data_block_length);
            break;
        case ScatterFunction::Add:
            scatter_neon_fp32<ScatterFunction::Add>(src, indices, dst, window, data_block_length);
            break;
        case ScatterFunction::Sub:
            scatter_neon_fp32<ScatterFunction::Sub>(src, indices, dst, window, data_block_length);
            break;
        case ScatterFunction::Max:
            scatter_neon_fp32<ScatterFunction::Max>(src, indices, dst, window, data_block_length);
            break;
        case ScatterFunction::Min:
            scatter_neon_fp32<ScatterFunction::Min>(src, indices, dst, window, data_block_length);
            break;
        default:
            ARM_COMPUTE_ERROR("Invalid reduction function for scatter.");
    }
    return;
}
} // namespace cpu
} // namespace arm_compute
