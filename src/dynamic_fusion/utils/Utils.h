/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_UTILS_UTILS
#define SRC_DYNAMIC_FUSION_UTILS_UTILS

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Pool2dAttributes.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Tensor should have backing memory. @ref MemoryType
 */
inline bool is_alloc_tensor(const ITensorInfo *tensor_info)
{
    return tensor_info->id() > ITensorInfo::invalid_tensor_id;
}

/** Tensor should not have backing memory. @ref MemoryType
 */
inline bool is_noalloc_tensor(const ITensorInfo *tensor_info)
{
    return tensor_info->id() < ITensorInfo::invalid_tensor_id;
}

/** @ref ITensorInfo has valid id
 */
inline bool is_valid_tensor(const ITensorInfo *tensor_info)
{
    return tensor_info->has_valid_id();
}

/** @ref ITensorInfo has invalid id
 */
inline bool is_invalid_tensor(const ITensorInfo *tensor_info)
{
    return !is_valid_tensor(tensor_info);
}

/** Inline function to convert @ref Pool2dAttributes to PoolingLayerInfo
*/
inline PoolingLayerInfo convert_pool_attr_to_pool_info(const Pool2dAttributes &pool_attr, bool mixed_precision = false, DataLayout data_layout = DataLayout::NHWC)
{
    // Create PadStrideInfo
    const Size2D        stride  = pool_attr.stride();
    const Padding2D     padding = pool_attr.pad();
    const PadStrideInfo pad_stride(stride.x(), stride.y(), padding.left, padding.top, arm_compute::DimensionRoundingType::FLOOR);

    return PoolingLayerInfo(pool_attr.pool_type(), pool_attr.pool_size(), data_layout, pad_stride, pool_attr.exclude_padding(), mixed_precision);
}
}
}
}

#endif /* SRC_DYNAMIC_FUSION_UTILS_UTILS */
