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

#ifndef ACL_ARM_COMPUTE_FUNCTION_INFO_SCATTERINFO_H
#define ACL_ARM_COMPUTE_FUNCTION_INFO_SCATTERINFO_H

#include "arm_compute/core/Error.h"

namespace arm_compute
{
/** Scatter Function */
enum class ScatterFunction
{
    Update = 0,
    Add    = 1,
    Sub    = 2,
    Max    = 3,
    Min    = 4
};
/** Scatter operator information */
struct ScatterInfo
{
    ScatterInfo(ScatterFunction f, bool zero) : func(f), zero_initialization(zero)
    {
        ARM_COMPUTE_ERROR_ON_MSG(f != ScatterFunction::Add && zero,
                                 "Zero initialisation is only supported with Add Scatter Function.");
    }
    ScatterFunction func;            /**< Type of scatter function to use with scatter operator*/
    bool zero_initialization{false}; /**< Fill output tensors with 0. Only available with add scatter function. */
};
} // namespace arm_compute

#endif // ACL_ARM_COMPUTE_FUNCTION_INFO_SCATTERINFO_H
