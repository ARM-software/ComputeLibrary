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
#include "ckw/Error.h"
#include "ckw/Kernel.h"
#include "ckw/TileOperand.h"
#include "src/Prototype.h"

namespace ckw
{

namespace
{

inline TensorComponentOperand &get_or_create_component(std::unique_ptr<TensorComponentOperand> &ptr, const ::std::string &name, TensorComponent component)
{
    if(ptr == nullptr)
    {
        ptr = std::make_unique<TensorComponentOperand>(name, component);
    }

    return *ptr;
}

} // namespace

// =================================================================================================
// TensorOperand
// =================================================================================================

TensorOperand::TensorOperand(const std::string &name, const TensorInfo &info)
    : OperandBase(name), _info(info)
{
}

prototype::Operand TensorOperand::create_impl_operand(prototype::IGpuKernelWriter *writer) const
{
    CKW_UNUSED(writer);
    return { name() };
}

const TensorInfo &TensorOperand::info() const
{
    return _info;
}

TensorInfo &TensorOperand::info()
{
    return _info;
}

DataType TensorOperand::data_type() const
{
    return _info.data_type();
}

bool TensorOperand::is_constant() const
{
    return false;
}

const TileOperand &TensorOperand::tile() const
{
    return *_tile;
}

TileOperand &TensorOperand::tile()
{
    return *_tile;
}

TensorOperand &TensorOperand::tile(TileOperand &tile)
{
    _tile = &tile;
    return *this;
}

const TensorTileSampler &TensorOperand::tile_sampler() const
{
    return _tile_sampler;
}

TensorTileSampler &TensorOperand::tile_sampler()
{
    return _tile_sampler;
}

TensorOperand &TensorOperand::tile_sampler(const TensorTileSampler &value)
{
    _tile_sampler = value;
    return *this;
}

TileOperand &TensorOperand::stride1()
{
    return get_or_create_component(_stride1, name(), TensorComponent::Stride1);
}

TileOperand &TensorOperand::stride2()
{
    return get_or_create_component(_stride2, name(), TensorComponent::Stride2);
}

TileOperand &TensorOperand::stride3()
{
    return get_or_create_component(_stride3, name(), TensorComponent::Stride3);
}

TileOperand &TensorOperand::stride4()
{
    return get_or_create_component(_stride4, name(), TensorComponent::Stride4);
}

TileOperand &TensorOperand::dim0()
{
    return get_or_create_component(_dim0, name(), TensorComponent::Dim0);
}

TileOperand &TensorOperand::dim1()
{
    return get_or_create_component(_dim1, name(), TensorComponent::Dim1);
}

TileOperand &TensorOperand::dim2()
{
    return get_or_create_component(_dim2, name(), TensorComponent::Dim2);
}

TileOperand &TensorOperand::dim3()
{
    return get_or_create_component(_dim3, name(), TensorComponent::Dim3);
}

TileOperand &TensorOperand::dim4()
{
    return get_or_create_component(_dim4, name(), TensorComponent::Dim4);
}

TileOperand &TensorOperand::dim1_dim2()
{
    return get_or_create_component(_dim1_dim2, name(), TensorComponent::Dim1xDim2);
}

TileOperand &TensorOperand::dim1_dim2_dim3()
{
    return get_or_create_component(_dim1_dim2_dim3, name(), TensorComponent::Dim1xDim2xDim3);
}

TileOperand &TensorOperand::offset_first_element_in_bytes()
{
    return get_or_create_component(_offset_first_element_in_bytes, name(), TensorComponent::OffsetFirstElement);
}

// =================================================================================================
// TensorComponentOperand
// =================================================================================================

TensorComponentOperand::TensorComponentOperand(const ::std::string &name, TensorComponent component)
    : TileOperand(name, DataType::Int32), _component(component)
{
}

prototype::Operand TensorComponentOperand::create_impl_operand(prototype::IGpuKernelWriter *writer) const
{
    CKW_UNUSED(writer);
    prototype::OperandType type{ prototype::OperandType::Unknown };

    switch(_component)
    {
        case TensorComponent::OffsetFirstElement:
            type = prototype::OperandType::TensorDataOffset;
            break;

        case TensorComponent::Stride1:
            type = prototype::OperandType::TensorStride1;
            break;

        case TensorComponent::Stride2:
            type = prototype::OperandType::TensorStride2;
            break;

        case TensorComponent::Stride3:
            type = prototype::OperandType::TensorStride3;
            break;

        case TensorComponent::Stride4:
            type = prototype::OperandType::TensorStride4;
            break;

        case TensorComponent::Dim0:
            type = prototype::OperandType::TensorDim0;
            break;

        case TensorComponent::Dim1:
            type = prototype::OperandType::TensorDim1;
            break;

        case TensorComponent::Dim2:
            type = prototype::OperandType::TensorDim2;
            break;

        case TensorComponent::Dim3:
            type = prototype::OperandType::TensorDim3;
            break;

        case TensorComponent::Dim4:
            type = prototype::OperandType::TensorDim4;
            break;

        case TensorComponent::Dim1xDim2:
            type = prototype::OperandType::TensorDim1xDim2;
            break;

        case TensorComponent::Dim1xDim2xDim3:
            type = prototype::OperandType::TensorDim1xDim2xDim3;
            break;

        default:
            CKW_ASSERT(false);
    }

    return prototype::Operand(name(), type);
}

} // namespace ckw
