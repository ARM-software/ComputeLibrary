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


namespace ckw
{
Tensor3dMapper::Tensor3dMapper(ITensor *tensor, TensorSampler sampler)
        : _tensor(tensor), _sampler(sampler)
{
}

std::string Tensor3dMapper::tensor_component_x() const
{
    const TensorSamplerFormat format = _sampler.format();
    switch(format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Dim0).scalar(0,0).str;
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return "";
    }
}

std::string Tensor3dMapper::tensor_component_y() const
{
    const TensorSamplerFormat format = _sampler.format();
    switch(format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
            return _tensor->component(TensorComponentType::Dim1xDim2).scalar(0,0).str;
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Dim1).scalar(0,0).str;
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return "";
    }
}

std::string Tensor3dMapper::tensor_component_z() const
{
    const TensorSamplerFormat format = _sampler.format();
    switch(format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
            return "1";
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Dim2).scalar(0,0).str;
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return "";
    }
}

std::string Tensor3dMapper::tensor_component_stride_x() const
{
    const TensorSamplerFormat format = _sampler.format();
    switch(format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Stride0).scalar(0,0).str;
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return "";
    }
}

std::string Tensor3dMapper::tensor_component_stride_y() const
{
    const TensorSamplerFormat format = _sampler.format();
    switch(format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Stride1).scalar(0,0).str;
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return "";
    }
}

std::string Tensor3dMapper::tensor_component_stride_z() const
{
    const TensorSamplerFormat format = _sampler.format();
    switch(format)
    {
        case TensorSamplerFormat::Dim0_Dim1xDim2_1:
            return "0";
        case TensorSamplerFormat::Dim0_Dim1_Dim2:
            return _tensor->component(TensorComponentType::Stride2).scalar(0,0).str;
        default:
            CKW_THROW_MSG("Unsupported tensor format");
            return "";
    }
}

std::string Tensor3dMapper::tensor_component_stride_batch() const
{
    return _tensor->component(TensorComponentType::Stride3).scalar(0,0).str;
}

TensorSampler Tensor3dMapper::sampler() const
{
    return _sampler;
}

ITensor *Tensor3dMapper::tensor() const
{
    return _tensor;
}
} // namespace ckw