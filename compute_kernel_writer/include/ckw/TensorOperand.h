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

#ifndef CKW_INCLUDE_CKW_TENSOROPERAND_H
#define CKW_INCLUDE_CKW_TENSOROPERAND_H

#include "ckw/TileOperand.h"

namespace ckw
{

class ITensor;
class TensorInfo;

/** A tensor operand provides access to the tensor info, tensor storages for load/store operations
 * and tensor components (e.g. shape, strides, etc.) in the form of @ref TileOperand objects.
 */
class TensorOperand
{
public:
    // _tensor field is completely hidden from the public API to avoid any misuse.
    // Only kernel writer class interacts with tensor operand hence we allow it to access this field.
    friend class KernelWriter;

    /** Create an empty tensor operand.
     *
     * The new tensor operand doesn't refer to any tensor therefore it is not useable.
     */
    TensorOperand();

    /** Check if the tensor operand contains a tensor and therefore useable. */
    bool is_valid() const;

    /** Get the tensor info. */
    const TensorInfo &info() const;

    /** Get the operand that contains the stride in dimension 0 of the tensor. */
    TileOperand stride0();

    /** Get the operand that contains the stride in dimension 1 of the tensor. */
    TileOperand stride1();

    /** Get the operand that contains the stride in dimension 2 of the tensor. */
    TileOperand stride2();

    /** Get the operand that contains the stride in dimension 3 of the tensor. */
    TileOperand stride3();

    /** Get the operand that contains the stride in dimension 4 of the tensor. */
    TileOperand stride4();

    /** Get the operand that contains the size of dimension 0 of the tensor. */
    TileOperand dim0();

    /** Get the operand that contains the size of dimension 1 of the tensor. */
    TileOperand dim1();

    /** Get the operand that contains the size of dimension 2 of the tensor. */
    TileOperand dim2();

    /** Get the operand that contains the size of dimension 3 of the tensor. */
    TileOperand dim3();

    /** Get the operand that contains the size of dimension 4 of the tensor. */
    TileOperand dim4();

    /** Get the operand that contains the size of dimensions 1 and 2 collapsed. */
    TileOperand dim1_dim2();

    /** Get the operand that contains the size of dimensions 1, 2 and 3 collapsed. */
    TileOperand dim1_dim2_dim3();

    /** Get the operand that contains the size of dimensions 2 and 3 collapsed. */
    TileOperand dim2_dim3();

    /** Get the operand that contains the offset in bytes to the first element. */
    TileOperand offset_first_element_in_bytes();

private:
    /** Initialize a new instance of @ref TensorOperand class for a tensor. */
    TensorOperand(ITensor &tensor);

    ITensor *_tensor;
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TENSOROPERAND_H
