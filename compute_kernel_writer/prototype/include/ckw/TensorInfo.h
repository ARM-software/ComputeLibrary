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

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_TENSORINFO_H
#define CKW_PROTOTYPE_INCLUDE_CKW_TENSORINFO_H

#include "ckw/Types.h"

#include <array>
#include <cstdint>

namespace ckw
{
/** Compute Kernel Writer tensor data layout (or memory format) */
enum class TensorDataLayout
{
    Unknown,
    Nhwc,
    Ndhwc
};

/** Compute Kernel Writer tensor data layout component */
enum class TensorDataLayoutComponent
{
    Unknown,
    N,
    D,
    H,
    W,
    C,
};

/** Compute Kernel Writer tensor component bitmask. The bitmask can be used to retrieve
 *  the info from @ref TensorComponent.
 */
enum class TensorComponentBitmask : uint32_t
{
    OffsetFirstElement = 0x01000000, // For example, OffsetFirstElement in @ref TensorComponent
    Stride             = 0x02000000, // For example, stride0 in @ref TensorComponent
    Dimension          = 0x04000000, // For example, Dim0 in @ref TensorComponent
    FoldedDimensions   = 0x08000000, // For example, Dim0xDim1 in @ref TensorComponent
};

/** Compute Kernel Writer tensor component. The tensor components are used to access specific backend-agnostic tensor arguments,
 *  such as the tensor dimensions and tensor strides.
 *  The data type is represented as an integer. The value of the integer value
 *  is assigned to retrieve the information through the @ref TensorComponentBitmask.
 */
enum class TensorComponent : uint32_t
{
    Unknown            = 0x00000000,
    OffsetFirstElement = 0x01000000,
    Stride0            = 0x02000001,
    Stride1            = 0x02000010,
    Stride2            = 0x02000100,
    Stride3            = 0x02001000,
    Stride4            = 0x02010000,
    Dim0               = 0x04000001,
    Dim1               = 0x04000010,
    Dim2               = 0x04000100,
    Dim3               = 0x04001000,
    Dim4               = 0x04010000,
    Dim1xDim2          = 0x08000110,
    Dim2xDim3          = 0x08001100,
    Dim1xDim2xDim3     = 0x08001110
};

/** Compute Kernel Writer tensor storage. The tensor storage represents the type of tensor memory object.
 */
enum class TensorStorage : uint32_t
{
    Unknown            = 0x00000000,
    BufferUint8Ptr     = 0x01000000,
    Texture2dReadOnly  = 0x02000001,
    Texture2dWriteOnly = 0x02000010,
};

/** Compute Kernel Writer tensor shape
 *  Negative dimensions can be interpreted as dynamic dimensions by the Compute Kernel Writer
 */
using TensorShape = std::array<int32_t, 5>;

/** Compute Kernel Writer tensor info */
class TensorInfo
{
public:
    /** Constructor
     *
     * @param[in] dt    Tensor data type
     * @param[in] shape Tensor shape
     * @param[in] dl    Tensor data layout
     * @param[in] id    Tensor id. The id is used to keep track of the bound user tensor. Through the id,
     *                  the user can know what tensor has been used by the Compute Kernel Writer.
     *                  Possible id values:
     *                  - greater than or equal to 0: bind a user specific tensors
     *                  - less than 0: bind a virtual tensor (tile)
     */
    TensorInfo(DataType dt, const TensorShape &shape, TensorDataLayout dl, int32_t id);

    /** Set shape */
    TensorInfo &shape(const TensorShape &shape);

    /** Get shape */
    TensorShape shape() const;

    /** Set data type */
    TensorInfo &data_type(DataType dt);

    /** Get data type */
    DataType data_type() const;

    /** Set data layout */
    TensorInfo &data_layout(TensorDataLayout dl);

    /** Get data layout */
    TensorDataLayout data_layout() const;

    /** Set id */
    TensorInfo &id(int32_t id);

    /** Get layout */
    int32_t id() const;

private:
    TensorShape      _shape{ { 0 } };
    DataType         _dt{ DataType::Unknown };
    TensorDataLayout _dl{ TensorDataLayout::Unknown };
    int32_t          _id{ -1 };
};
} // namespace ckw

#endif /* CKW_PROTOTYPE_INCLUDE_CKW_TENSORINFO_H */
