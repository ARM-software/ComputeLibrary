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

#ifndef CKW_INCLUDE_CKW_TYPES_TENSORCOMPONENTTYPE_H
#define CKW_INCLUDE_CKW_TYPES_TENSORCOMPONENTTYPE_H

#include <cstdint>

namespace ckw
{

/** Compute Kernel Writer tensor component.
 *
 * The tensor components are used to access specific backend-agnostic tensor arguments,
 * such as the tensor dimensions and tensor strides.
 * The tensor component is represented as an unsigned integer. The value of the integer value
 * is assigned to retrieve the information through the @ref TensorComponentBitmask.
 */
enum class TensorComponentType : uint32_t
{
    Unknown            = 0x00000000,
    OffsetFirstElement = 0x01000000,
    Stride0            = 0x02000001,
    Stride1            = 0x02000002,
    Stride2            = 0x02000003,
    Stride3            = 0x02000004,
    Stride4            = 0x02000005,
    Dim0               = 0x04000001,
    Dim1               = 0x04000002,
    Dim2               = 0x04000003,
    Dim3               = 0x04000004,
    Dim4               = 0x04000005,
    Dim1xDim2          = 0x08000032,
    Dim2xDim3          = 0x08000043,
    Dim1xDim2xDim3     = 0x08000432
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TYPES_TENSORCOMPONENTTYPE_H
