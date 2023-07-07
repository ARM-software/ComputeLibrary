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

#ifndef COMPUTE_KERNEL_WRITER_INCLUDE_CKW_TENSORINFO_H
#define COMPUTE_KERNEL_WRITER_INCLUDE_CKW_TENSORINFO_H

#include "ckw/types/DataType.h"
#include "ckw/types/TensorDataLayout.h"
#include <array>
#include <cstdint>

namespace ckw
{

/** Compute Kernel Writer tensor shape
 *  The value -1 for the tensor dimension is reserved to dynamic dimensions.
 */
using TensorShape = std::array<int32_t, 5>;

/** Tensor dimension value reserved to dynamic dimensions */
constexpr int32_t kDynamicTensorDimensionValue = -1;

/** Compute Kernel Writer tensor info */
class TensorInfo
{
public:
    /** Default constructor */
    TensorInfo() = default;
    /** Constructor
     *
     * @param[in] dt    Tensor data type
     * @param[in] shape Tensor shape
     * @param[in] dl    Tensor data layout
     * @param[in] id    Tensor id. The id is used to keep track of the user tensor binded. Through the id,
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

#endif /* COMPUTE_KERNEL_WRITER_INCLUDE_CKW_TENSORINFO_H */
