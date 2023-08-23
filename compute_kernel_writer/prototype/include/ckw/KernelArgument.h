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

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_KERNELARGUMENT_H
#define CKW_PROTOTYPE_INCLUDE_CKW_KERNELARGUMENT_H

#include "ckw/TensorInfo.h"
#include <cstdint>

namespace ckw
{

class TensorOperand;
class TensorComponentOperand;

/** A kernel argument which can be either a tensor storage or a tensor component. */
class KernelArgument
{
public:
    /** The type of kernel argument. */
    enum class Type : int32_t
    {
        /** The argument that provides the read and/or write access to the tensor data.
         *
         * See @ref ckw::TensorStorage to see the list of supported storage type.
         */
        TensorStorage,

        /** The argument that provides extra information about the tensor.
         *
         * See @ref ckw::TensorComponent to see the list of supported component.
         */
        TensorComponent,
    };

    /** Initialize a new instance of kernel argument class for a tensor storage argument.
     *
     * @param[in] tensor The tensor whose storage is exposed to kernel arguments.
     */
    KernelArgument(TensorOperand &tensor);

    /** Initialize a new instance of kernel argument class for a tensor component argument.
     *
     * @param[in] tensor_component The tensor component to be exposed to kernel arguments.
     */
    KernelArgument(TensorComponentOperand &tensor_component);

    /** Get the type of kernel argument. */
    Type type() const;

    /** Get the argument ID.
     *
     * This method can be used to get the tensor info ID of both tensor storage and tensor component arguments.
     */
    int32_t id() const;

    /** Get the type of tensor storage.
     *
     * This method can only be used for tensor storage argument.
     */
    TensorStorageType tensor_storage_type() const;

    /** Get the tensor component type.
     *
     * This method can only be used for tensor component argument.
     */
    TensorComponentType tensor_component_type() const;

private:
    Type    _type;
    int32_t _id;

    union SubId
    {
        int32_t             unknown;
        TensorStorageType   tensor_storage_type;
        TensorComponentType tensor_component_type;
    };

    SubId _sub_id{ 0 };
};

} // namespace ckw

#endif // CKW_PROTOTYPE_INCLUDE_CKW_KERNELARGUMENT_H
