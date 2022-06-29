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

#include "input_transform.hpp"
#include "winograd_implementations.hpp"

#include <memory>
#include <string>

namespace arm_conv {
namespace winograd {
namespace input_transform {

#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
void sve_fp32_6x6(unsigned int, const float *, size_t, size_t, float *, size_t);
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
void a64_fp32_6x6(unsigned int, const float *, size_t, size_t, float *, size_t);
#else  // defined(__aarch64__)
void arm_fp32_6x6(unsigned int, const float *, size_t, size_t, float *, size_t);
#endif  // defined(__aarch64__)
void arm_fp32_4x4(unsigned int, const float *, size_t, size_t, float *, size_t);
void arm_fp32_1x8(unsigned int, const float *, size_t, size_t, float *, size_t);

#define IMPL(HEIGHT, WIDTH, FUNC, DRIVER) new Transform ## DRIVER <float, float>(#FUNC, HEIGHT, WIDTH, FUNC)

static const TransformImplementation<float> transforms_fp32[] = {
#if defined(__aarch64__)
#if defined(ARM_COMPUTE_ENABLE_SVE)
  { IMPL(6, 6, sve_fp32_6x6, Unpadded), MethodConstraints::RequiresSVE },
#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
  { IMPL(6, 6, a64_fp32_6x6, Unpadded) },
#else  // defined(__aarch64__)
  { IMPL(6, 6, arm_fp32_6x6, Unpadded) },
#endif  // defined(__aarch64__)
  { IMPL(4, 4, arm_fp32_4x4, Unpadded) },
  { IMPL(1, 8, arm_fp32_1x8, Unpadded) },
  { new TransformUnpadded<float, float>("arm_fp32_1x8", 8, 1, TransformUnpadded<float, float>::get_transposed_kernel(arm_fp32_1x8)) },
  { nullptr },
};

template <>
const TransformImplementation<float> *implementation_list(void)
{
  return transforms_fp32;
}

}  // namespace input_transform
}  // namespace winograd
}  // namespace arm_conv
