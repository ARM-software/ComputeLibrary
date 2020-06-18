/*
 * Copyright (c) 2020 ARM Limited.
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
#ifndef ARM_COMPUTE_EXPERIMENTAL_TYPES_H
#define ARM_COMPUTE_EXPERIMENTAL_TYPES_H

#include "arm_compute/core/TensorShape.h"

#include <map>
#include <vector>

namespace arm_compute
{
class ITensor;

/** Memory type */
enum class TensorType
{
    ACL_UNKNOWN = -1,
    ACL_SRC     = 0,
    ACL_SRC_0   = 0,
    ACL_SRC_1   = 1,
    ACL_SRC_2   = 2,
    ACL_DST     = 30,
    ACL_DST_0   = 30,
    ACL_DST_1   = 31,
    ACL_INT     = 50,
    ACL_INT_0   = 50,
    ACL_INT_1   = 51,
    ACL_INT_2   = 52
};

using InputTensorMap    = std::map<TensorType, const ITensor *>;
using OutputTensorMap   = std::map<TensorType, ITensor *>;
using OperatorTensorMap = OutputTensorMap;

namespace experimental
{
struct MemoryInfo
{
    MemoryInfo(TensorType type, size_t size, size_t alignment)
        : type(type), size(size), alignment(alignment)
    {
    }
    TensorType type;
    size_t     size;
    size_t     alignment;
};

using MemoryRequirements = std::vector<MemoryInfo>;
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_EXPERIMENTAL_TYPES_H */
