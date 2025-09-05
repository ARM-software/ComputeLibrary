/*
 * Copyright (c) 2025 Arm Limited.
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

#ifndef ACL_ARM_COMPUTE_CORE_CL_CL_DEFINITIONS_H
#define ACL_ARM_COMPUTE_CORE_CL_CL_DEFINITIONS_H

#include "include/CL/opencl.hpp"

#define CL_DEVICE_MATRIX_MULTIPLY_FP16_WITH_FP16_ACCUMULATORS_ARM (1ULL << 0)
#define CL_DEVICE_MATRIX_MULTIPLY_CAPABILITIES_ARM                0x41F4

namespace cl
{
namespace detail
{
#ifdef CL_DEVICE_MATRIX_MULTIPLY_CAPABILITIES_ARM
CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_MATRIX_MULTIPLY_CAPABILITIES_ARM, cl_ulong)
#endif // CL_DEVICE_MATRIX_MULTIPLY_CAPABILITIES_ARM
} // namespace detail
} // namespace cl
#endif // ACL_ARM_COMPUTE_CORE_CL_CL_DEFINITIONS_H
