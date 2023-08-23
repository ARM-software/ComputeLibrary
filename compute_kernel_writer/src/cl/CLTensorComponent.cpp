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

#include "src/cl/CLTensorComponent.h"
#include "ckw/Error.h"
#include "ckw/types/TensorComponentType.h"
#include "src/cl/CLTensorArgument.h"
#include "src/cl/CLTile.h"

namespace ckw
{

namespace
{

std::string create_component_name(const std::string &name, TensorComponentType x)
{
    std::string var_name(name);

    switch(x)
    {
        case TensorComponentType::OffsetFirstElement:
            var_name += "_offset_first_element";
            break;
        case TensorComponentType::Stride0:
            var_name += "_stride0";
            break;
        case TensorComponentType::Stride1:
            var_name += "_stride1";
            break;
        case TensorComponentType::Stride2:
            var_name += "_stride2";
            break;
        case TensorComponentType::Stride3:
            var_name += "_stride3";
            break;
        case TensorComponentType::Stride4:
            var_name += "_stride4";
            break;
        case TensorComponentType::Dim0:
            var_name += "_dim0";
            break;
        case TensorComponentType::Dim1:
            var_name += "_dim1";
            break;
        case TensorComponentType::Dim2:
            var_name += "_dim2";
            break;
        case TensorComponentType::Dim3:
            var_name += "_dim3";
            break;
        case TensorComponentType::Dim4:
            var_name += "_dim4";
            break;
        case TensorComponentType::Dim1xDim2:
            var_name += "_dim1xdim2";
            break;
        case TensorComponentType::Dim2xDim3:
            var_name += "_dim2xdim3";
            break;
        case TensorComponentType::Dim1xDim2xDim3:
            var_name += "_dim1xdim2xdim3";
            break;
        default:
            CKW_THROW_MSG("Unsupported tensor component");
            return "";
    }

    return var_name;
}

} // namespace

CLTensorComponent::CLTensorComponent(const CLTensorArgument &tensor, TensorComponentType component_type)
    : CLTile(create_component_name(tensor.name(), component_type), TileInfo(DataType::Int32)), _component_type(component_type)
{
}

CLTensorComponent::CLTensorComponent(const CLTensorArgument &tensor, TensorComponentType component_type, int32_t value)
    : CLTile({ { std::to_string(value) } }, DataType::Int32), _component_type(component_type)
{
    CKW_UNUSED(tensor);
}

CLTensorComponent::~CLTensorComponent() = default;

ITile &CLTensorComponent::tile()
{
    return *this;
}

const ITile &CLTensorComponent::tile() const
{
    return *this;
}

TensorComponentType CLTensorComponent::component_type() const
{
    return _component_type;
}

} // namespace ckw
