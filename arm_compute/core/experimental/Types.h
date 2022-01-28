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
#ifndef ARM_COMPUTE_EXPERIMENTAL_TYPES_H
#define ARM_COMPUTE_EXPERIMENTAL_TYPES_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorShape.h"

#include <vector>

namespace arm_compute
{
// Forward declaration
class ITensor;

/** Memory type */
enum TensorType : int32_t
{
    ACL_UNKNOWN = -1,
    ACL_SRC_DST = 0,

    // Src
    ACL_SRC     = 0,
    ACL_SRC_0   = 0,
    ACL_SRC_1   = 1,
    ACL_SRC_2   = 2,
    ACL_SRC_3   = 3,
    ACL_SRC_4   = 4,
    ACL_SRC_5   = 5,
    ACL_SRC_6   = 6,
    ACL_SRC_END = 6,

    // Dst
    ACL_DST     = 30,
    ACL_DST_0   = 30,
    ACL_DST_1   = 31,
    ACL_DST_2   = 32,
    ACL_DST_END = 32,

    // Aux
    ACL_INT     = 50,
    ACL_INT_0   = 50,
    ACL_INT_1   = 51,
    ACL_INT_2   = 52,
    ACL_INT_3   = 53,
    ACL_INT_4   = 54,
    ACL_SRC_VEC = 256,
    ACL_DST_VEC = 512,
    ACL_INT_VEC = 1024,

    // Aliasing Types
    // Conv etc
    ACL_BIAS = ACL_SRC_2,

    // Gemm
    ACL_VEC_ROW_SUM = ACL_SRC_3,
    ACL_VEC_COL_SUM = ACL_SRC_4,
    ACL_SHIFTS      = ACL_SRC_5,
    ACL_MULTIPLIERS = ACL_SRC_6,

    // (EXPERIMENTAL_POST_OPS) Post ops arguments begin after everything else
    EXPERIMENTAL_ACL_POST_OP_ARG       = 2048,
    EXPERIMENTAL_ACL_POST_OP_ARG_FIRST = EXPERIMENTAL_ACL_POST_OP_ARG,
    EXPERIMENTAL_ACL_POST_OP_ARG_LAST  = EXPERIMENTAL_ACL_POST_OP_ARG_FIRST + 1024, // Max number of post op arguments
};

namespace experimental
{
enum class MemoryLifetime
{
    Temporary  = 0,
    Persistent = 1,
    Prepare    = 2,
};
struct MemoryInfo
{
    MemoryInfo() = default;

    MemoryInfo(int slot, size_t size, size_t alignment = 0) noexcept
        : slot(slot),
          size(size),
          alignment(alignment)
    {
    }

    MemoryInfo(int slot, MemoryLifetime lifetime, size_t size, size_t alignment = 0) noexcept
        : slot(slot),
          lifetime(lifetime),
          size(size),
          alignment(alignment)
    {
    }

    bool merge(int slot, size_t new_size, size_t new_alignment = 0) noexcept
    {
        if(slot != this->slot)
        {
            return false;
        }

        size      = std::max(size, new_size);
        alignment = std::max(alignment, new_alignment);

        return true;
    }

    int            slot{ ACL_UNKNOWN };
    MemoryLifetime lifetime{ MemoryLifetime::Temporary };
    size_t         size{ 0 };
    size_t         alignment{ 64 };
};

using MemoryRequirements = std::vector<MemoryInfo>;
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_EXPERIMENTAL_TYPES_H */
