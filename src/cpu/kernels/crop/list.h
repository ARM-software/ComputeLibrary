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
#ifndef SRC_CORE_NEON_KERNELS_CROP_LIST_H
#define SRC_CORE_NEON_KERNELS_CROP_LIST_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/Registrars.h"
#include "src/cpu/kernels/crop/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
#define DECLARE_CROP_KERNEL(func_name)                                                                       \
    void func_name(const ITensor *input, const ITensor *output, float *output_ptr, Coordinates input_offset, \
                   int32_t window_step_x, int32_t output_width_start, int32_t output_width_limit, bool input_has_single_channel, bool is_width_flipped)

DECLARE_CROP_KERNEL(fp16_in_bounds_crop_window);
DECLARE_CROP_KERNEL(fp32_in_bounds_crop_window);
DECLARE_CROP_KERNEL(s8_in_bounds_crop_window);
DECLARE_CROP_KERNEL(s16_in_bounds_crop_window);
DECLARE_CROP_KERNEL(s32_in_bounds_crop_window);
DECLARE_CROP_KERNEL(u8_in_bounds_crop_window);
DECLARE_CROP_KERNEL(u16_in_bounds_crop_window);
DECLARE_CROP_KERNEL(u32_in_bounds_crop_window);

#undef DECLARE_CROP_KERNEL

} // namespace cpu
} // namespace arm_compute
#endif //SRC_CORE_NEON_KERNELS_CROP_LIST_H
