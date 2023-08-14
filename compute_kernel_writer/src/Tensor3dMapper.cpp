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

#include "Tensor3dMapper.h"

#include "ckw/Error.h"
#include "ckw/types/TensorSamplerTypes.h"
#include "src/ITensor.h"
#include "src/ITile.h"

namespace ckw
{
Tensor3dMapper::Tensor3dMapper(ITensor *tensor, TensorSamplerFormat format)
        : _tensor(tensor), _format(format)
{
}

TileVariable Tensor3dMapper::dim_x() const
{
    switch(_format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Dim0).scalar(0, 0);
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return _tensor->component(TensorComponentType::Unknown).scalar(0, 0);
    }
}

TileVariable Tensor3dMapper::dim_y() const
{
    switch(_format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
            return _tensor->component(TensorComponentType::Dim1xDim2).scalar(0, 0);
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Dim1).scalar(0, 0);
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return _tensor->component(TensorComponentType::Unknown).scalar(0, 0);
    }
}

TileVariable Tensor3dMapper::dim_z() const
{
    TileVariable dim_one;

    switch(_format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
            dim_one = _tensor->component(TensorComponentType::Dim3).scalar(0, 0);
            dim_one.str = "1";
            return dim_one;
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Dim2).scalar(0, 0);
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return _tensor->component(TensorComponentType::Unknown).scalar(0, 0);
    }
}

TileVariable Tensor3dMapper::dim_batch() const
{
    TileVariable dim_one;

    switch(_format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Dim3).scalar(0, 0);
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return _tensor->component(TensorComponentType::Unknown).scalar(0, 0);
    }
}

TileVariable Tensor3dMapper::stride_x() const
{
    switch(_format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Stride0).scalar(0, 0);
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return _tensor->component(TensorComponentType::Unknown).scalar(0, 0);
    }
}

TileVariable Tensor3dMapper::stride_y() const
{
    switch(_format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Stride1).scalar(0, 0);
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return _tensor->component(TensorComponentType::Unknown).scalar(0, 0);
    }
}

TileVariable Tensor3dMapper::stride_z() const
{
    TileVariable stride_zero;

    switch(_format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
            stride_zero = _tensor->component(TensorComponentType::Stride3).scalar(0, 0);
            stride_zero.str = "0";
            return stride_zero;
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Stride2).scalar(0, 0);
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return _tensor->component(TensorComponentType::Unknown).scalar(0, 0);
    }
}

TileVariable Tensor3dMapper::stride_batch() const
{
    switch(_format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Stride3).scalar(0, 0);
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return _tensor->component(TensorComponentType::Unknown).scalar(0, 0);
    }
}
} // namespace ckw