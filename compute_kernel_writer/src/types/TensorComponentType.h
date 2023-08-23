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

#ifndef CKW_SRC_TYPES_TENSORCOMPONENTTYPE_H
#define CKW_SRC_TYPES_TENSORCOMPONENTTYPE_H

#include <cstdint>

namespace ckw
{

/** Compute Kernel Writer tensor component bitmask.
 *
 * The bitmask can be used to retrieve the info from @ref TensorComponent.
 */
enum class TensorComponentBitmask : uint32_t
{
    OffsetFirstElement = 0x01000000, // For example, OffsetFirstElement in TensorComponent
    Stride             = 0x02000000, // For example, stride0 in TensorComponent
    Dimension          = 0x04000000, // For example, Dim0 in TensorComponent
    FoldedDimensions   = 0x08000000, // For example, Dim0xDim1 in TensorComponent
};

/** Mask to retrieve the component index (for example, 1 for stride1, 2 for stride2, or 1 and 2 for Dim1xDim2).
 *
 * The 4 least significant half-bytes (nibbles) of the @ref TensorComponent are used to retrieve the specific component index.
 * TensorComponent = | i7 | i6 | i5 | i4 | i3 | i2 | i1 | i0 |, where i7,...i0 are the nibbles
 * of the TensorComponent hexadecimal number. i0, i1, i2 and i3 are reserved to the component index.
 *
 * In particular:
 *
 *   -# i0: reserved to the first folded dimension component index
 *   -# i1: reserved to the second folded dimension component index
 *   -# i2: reserved to the third folded dimension component index
 *   -# i3: reserved to the fourth folded dimension component index
 *
 * Therefore, if there are no folded dimensions (dimensions and strides), only i0 is used.
 * Instead, if there are two folded dimensions, only i0 and i1 are used.
 *
 * The component index is stored with the corresponding hexadecimal number + 1,
 * hence the component index 0 is represented as 1, while the component index 3 is represented as 4.
 */
enum class TensorComponentIndexBitmask : uint32_t
{
    All    = 0x0000ffff, // All nibbles reserved to the tensor component index
    Index0 = 0x0000000f, // Folded dimension 0
    Index1 = 0x000000f0, // Folded dimension 1
    Index2 = 0x00000f00, // Folded dimension 2
    Index3 = 0x0000f000  // Folded dimension 3
};

/** The maximum number of folded dimensions. */
constexpr int tensor_component_index_max_count = 4;

} // namespace ckw

#endif // CKW_SRC_TYPES_TENSORCOMPONENTTYPE_H
