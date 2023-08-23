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
#include "ckw/TensorInfo.h"
#include "ckw/TileOperand.h"
#include "src/Prototype.h"

namespace ckw
{

namespace
{

TensorComponentOperand &get_or_create_component(TensorOperand &tensor, std::unique_ptr<TensorComponentOperand> &ptr, TensorComponentType component)
{
    if(ptr == nullptr)
    {
        ptr = std::make_unique<TensorComponentOperand>(tensor, component);
    }

    return *ptr;
}

} // namespace

// =================================================================================================
// TensorOperand
// =================================================================================================

TensorOperand::TensorOperand(const std::string &name, const TensorInfo &info, TensorStorageType storage_type)
    : OperandBase(name), _info(info), _storage_type(storage_type)
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

TensorStorageType TensorOperand::storage_type() const
{
    return _storage_type;
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

TensorComponentOperand &TensorOperand::stride1()
{
    return get_or_create_component(*this, _stride1, TensorComponentType::Stride1);
}

TensorComponentOperand &TensorOperand::stride2()
{
    return get_or_create_component(*this, _stride2, TensorComponentType::Stride2);
}

TensorComponentOperand &TensorOperand::stride3()
{
    return get_or_create_component(*this, _stride3, TensorComponentType::Stride3);
}

TensorComponentOperand &TensorOperand::stride4()
{
    return get_or_create_component(*this, _stride4, TensorComponentType::Stride4);
}

TensorComponentOperand &TensorOperand::dim0()
{
    return get_or_create_component(*this, _dim0, TensorComponentType::Dim0);
}

TensorComponentOperand &TensorOperand::dim1()
{
    return get_or_create_component(*this, _dim1, TensorComponentType::Dim1);
}

TensorComponentOperand &TensorOperand::dim2()
{
    return get_or_create_component(*this, _dim2, TensorComponentType::Dim2);
}

TensorComponentOperand &TensorOperand::dim3()
{
    return get_or_create_component(*this, _dim3, TensorComponentType::Dim3);
}

TensorComponentOperand &TensorOperand::dim4()
{
    return get_or_create_component(*this, _dim4, TensorComponentType::Dim4);
}

TensorComponentOperand &TensorOperand::dim1_dim2()
{
    return get_or_create_component(*this, _dim1_dim2, TensorComponentType::Dim1xDim2);
}

TensorComponentOperand &TensorOperand::dim1_dim2_dim3()
{
    return get_or_create_component(*this, _dim1_dim2_dim3, TensorComponentType::Dim1xDim2xDim3);
}

TensorComponentOperand &TensorOperand::offset_first_element_in_bytes()
{
    return get_or_create_component(*this, _offset_first_element_in_bytes, TensorComponentType::OffsetFirstElement);
}

// =================================================================================================
// TensorComponentOperand
// =================================================================================================

TensorComponentOperand::TensorComponentOperand(TensorOperand &tensor, TensorComponentType component)
    : TileOperand(tensor.name(), DataType::Int32), _tensor(tensor), _component(component)
{
}

TensorOperand &TensorComponentOperand::tensor()
{
    return _tensor;
}

const TensorOperand &TensorComponentOperand::tensor() const
{
    return _tensor;
}

TensorComponentType TensorComponentOperand::component_type() const
{
    return _component;
}

prototype::Operand TensorComponentOperand::create_impl_operand(prototype::IGpuKernelWriter *writer) const
{
    CKW_UNUSED(writer);
    prototype::OperandType type{ prototype::OperandType::Unknown };

    switch(_component)
    {
        case TensorComponentType::OffsetFirstElement:
            type = prototype::OperandType::TensorDataOffset;
            break;

        case TensorComponentType::Stride1:
            type = prototype::OperandType::TensorStride1;
            break;

        case TensorComponentType::Stride2:
            type = prototype::OperandType::TensorStride2;
            break;

        case TensorComponentType::Stride3:
            type = prototype::OperandType::TensorStride3;
            break;

        case TensorComponentType::Stride4:
            type = prototype::OperandType::TensorStride4;
            break;

        case TensorComponentType::Dim0:
            type = prototype::OperandType::TensorDim0;
            break;

        case TensorComponentType::Dim1:
            type = prototype::OperandType::TensorDim1;
            break;

        case TensorComponentType::Dim2:
            type = prototype::OperandType::TensorDim2;
            break;

        case TensorComponentType::Dim3:
            type = prototype::OperandType::TensorDim3;
            break;

        case TensorComponentType::Dim4:
            type = prototype::OperandType::TensorDim4;
            break;

        case TensorComponentType::Dim1xDim2:
            type = prototype::OperandType::TensorDim1xDim2;
            break;

        case TensorComponentType::Dim1xDim2xDim3:
            type = prototype::OperandType::TensorDim1xDim2xDim3;
            break;

        default:
            CKW_ASSERT(false);
    }

    return prototype::Operand(name(), type);
}

} // namespace ckw
