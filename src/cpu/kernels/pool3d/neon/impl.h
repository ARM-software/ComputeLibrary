/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_CORE_POOLING_3D_LAYER_IMPL_H
#define SRC_CORE_POOLING_3D_LAYER_IMPL_H

#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
class Window;
struct Pooling3dLayerInfo;
namespace cpu
{
template <typename T>
void poolingMxNxD_fp_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info, const Window &window);

template <typename T>
void poolingMxNxD_q8_neon_ndhwc(const ITensor *src, ITensor *dst0, Pooling3dLayerInfo &pool_info, const Window &window);
} // namespace cpu
} // namespace arm_compute
#endif //define SRC_CORE_POOLING_3D_LAYER_IMPL_H
