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

#include "ckw/TensorOperand.h"
#include "src/ITensor.h"

namespace ckw
{

TensorOperand::TensorOperand(ITensor &tensor)
    : _tensor(tensor)
{
}

const TensorInfo &TensorOperand::info() const
{
    return _tensor.info();
}

TileOperand TensorOperand::stride0()
{
    return TileOperand(_tensor.component(TensorComponentType::Stride0));
}

TileOperand TensorOperand::stride1()
{
    return TileOperand(_tensor.component(TensorComponentType::Stride1));
}

TileOperand TensorOperand::stride2()
{
    return TileOperand(_tensor.component(TensorComponentType::Stride2));
}

TileOperand TensorOperand::stride3()
{
    return TileOperand(_tensor.component(TensorComponentType::Stride3));
}

TileOperand TensorOperand::stride4()
{
    return TileOperand(_tensor.component(TensorComponentType::Stride4));
}

TileOperand TensorOperand::dim0()
{
    return TileOperand(_tensor.component(TensorComponentType::Dim0));
}

TileOperand TensorOperand::dim1()
{
    return TileOperand(_tensor.component(TensorComponentType::Dim1));
}

TileOperand TensorOperand::dim2()
{
    return TileOperand(_tensor.component(TensorComponentType::Dim2));
}

TileOperand TensorOperand::dim3()
{
    return TileOperand(_tensor.component(TensorComponentType::Dim3));
}

TileOperand TensorOperand::dim4()
{
    return TileOperand(_tensor.component(TensorComponentType::Dim4));
}

TileOperand TensorOperand::dim1_dim2()
{
    return TileOperand(_tensor.component(TensorComponentType::Dim1xDim2));
}

TileOperand TensorOperand::dim1_dim2_dim3()
{
    return TileOperand(_tensor.component(TensorComponentType::Dim1xDim2xDim3));
}

TileOperand TensorOperand::dim2_dim3()
{
    return TileOperand(_tensor.component(TensorComponentType::Dim2xDim3));
}

TileOperand TensorOperand::offset_first_element_in_bytes()
{
    return TileOperand(_tensor.component(TensorComponentType::OffsetFirstElement));
}

} // namespace ckw