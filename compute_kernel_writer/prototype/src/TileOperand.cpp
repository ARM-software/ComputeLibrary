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

#include "ckw/TileOperand.h"
#include "ckw/Error.h"
#include "src/Prototype.h"

namespace ckw
{

TileOperand::TileOperand(const std::string &name, const TileInfo &info)
    : OperandBase(name),
      _info(info),
      _value{ std::vector<std::string>{ "0" } },
      _constant(false)
{
}

TileOperand::TileOperand(const std::string &name, DataType data_type)
    : OperandBase(name),
      _info(TileInfo{ data_type }),
      _value{ std::vector<std::string>{ "0" } },
      _constant(false)
{
}

TileOperand::TileOperand(const std::string &name, int32_t value)
    : OperandBase(name),
      _info(TileInfo{ DataType::Int32 }),
      _value{ std::vector<std::string>{ std::to_string(value) } },
      _constant(true)
{
}

TileOperand::TileOperand(const std::string &name, float value)
    : OperandBase(name),
      _info(TileInfo{ DataType::Fp32 }),
      _value{ std::vector<std::string>{ std::to_string(value) } },
      _constant(true)
{
}

TileOperand::TileOperand(const std::string &name, const TileContainer &vals, DataType dt)
    : OperandBase(name),
      _info(TileInfo{ dt, static_cast<int32_t>(vals.size()), static_cast<int32_t>(vals[0].size()) }),
      _value(vals),
      _constant(true)
{
}

prototype::Operand TileOperand::create_impl_operand(prototype::IGpuKernelWriter *writer) const
{
    CKW_UNUSED(writer);

    if(_constant)
    {
        if(is_scalar())
        {
            switch(_info.data_type())
            {
                case DataType::Int32:
                    return prototype::Operand(_value[0][0], prototype::OperandType::ScalarInt32);

                case DataType::Fp32:
                    return prototype::Operand(_value[0][0], prototype::OperandType::ScalarFp32);

                default:
                    CKW_ASSERT(false);
            }
        }
        else
        {
            return prototype::Operand(name());
        }
    }
    else
    {
        return prototype::Operand(name(), prototype::OperandType::Tile);
    }
}

const TileInfo &TileOperand::tile_info() const
{
    return _info;
}

DataType TileOperand::data_type() const
{
    return _info.data_type();
}

bool TileOperand::is_constant() const
{
    return _constant;
}

bool TileOperand::is_scalar() const
{
    return _info.width() == 1 && _info.height() == 1;
}

std::string TileOperand::scalar_value() const
{
    CKW_ASSERT(is_scalar());
    CKW_ASSERT(is_constant());

    return _value[0][0];
}

const TileContainer &TileOperand::value() const
{
    return _value;
}

} // namespace ckw
