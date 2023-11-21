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

#include "ckw/KernelArgument.h"

#include "ckw/Error.h"
#include "ckw/TensorOperand.h"

namespace ckw
{

KernelArgument::KernelArgument(TensorOperand &tensor) : _type(Type::TensorStorage), _id(tensor.info().id())
{
    _sub_id.tensor_storage_type = tensor.storage_type();
}

KernelArgument::KernelArgument(TensorComponentOperand &tensor_component)
    : _type(Type::TensorComponent), _id(tensor_component.tensor().info().id())
{
    _sub_id.tensor_component_type = tensor_component.component_type();
}

KernelArgument::Type KernelArgument::type() const
{
    return _type;
}

int32_t KernelArgument::id() const
{
    return _id;
}

TensorStorageType KernelArgument::tensor_storage_type() const
{
    CKW_ASSERT(_type == Type::TensorStorage);
    return _sub_id.tensor_storage_type;
}

TensorComponentType KernelArgument::tensor_component_type() const
{
    CKW_ASSERT(_type == Type::TensorComponent);
    return _sub_id.tensor_component_type;
}

} // namespace ckw
