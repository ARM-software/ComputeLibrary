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

#include "ckw/types/ConstantData.h"

#include <limits>

namespace ckw
{
namespace
{
    template<typename T>
    inline typename std::enable_if<std::is_same<T, float>::value, std::string>::type to_str(T value)
    {
        std::stringstream ss;
        ss << std::scientific << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
        return ss.str();
    }

    template<typename T>
    inline typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, bool>::value, std::string>::type to_str(T value)
    {
        return std::to_string(value);
    }

    template<typename T>
    inline typename std::enable_if<std::is_same<T, bool>::value, std::string>::type to_str(T value)
    {
        return std::to_string((int) value);
    }
}

template<typename T>
ConstantData::ConstantData(std::initializer_list<std::initializer_list<T>> values, DataType data_type)
    : _data_type(data_type)
{
    CKW_ASSERT(validate<T>(data_type));
    CKW_ASSERT(values.size() > 0);

    for(auto value_arr: values)
    {
        // Each row must have the same number of elements
        CKW_ASSERT(value_arr.size() == (*values.begin()).size());

        StringVector vec;
        std::transform(value_arr.begin(), value_arr.end(),
            std::back_inserter(vec),
            [](T val) { return to_str(val); });

        _values.push_back(std::move(vec));
    }
}

template<typename T>
bool ConstantData::validate(DataType data_type)
{
    switch(data_type)
    {
        case DataType::Fp32:
        case DataType::Fp16:
            return std::is_same<T, float>::value;
        case DataType::Bool:
            return std::is_same<T, bool>::value;
        case DataType::Int32:
        case DataType::Int16:
        case DataType::Int8:
            return std::is_same<T, int32_t>::value;
        case DataType::Uint32:
        case DataType::Uint16:
        case DataType::Uint8:
            return std::is_same<T, uint32_t>::value;
        default:
            CKW_THROW_MSG("Unknown data type!");
            break;
    }
}

// Necessary instantiations for compiler to recognize
template ConstantData::ConstantData(std::initializer_list<std::initializer_list<int32_t>>, DataType);
template ConstantData::ConstantData(std::initializer_list<std::initializer_list<uint32_t>>, DataType);
template ConstantData::ConstantData(std::initializer_list<std::initializer_list<bool>>, DataType);
template ConstantData::ConstantData(std::initializer_list<std::initializer_list<float>>, DataType);

template bool ConstantData::validate<int32_t>(DataType);
template bool ConstantData::validate<uint32_t>(DataType);
template bool ConstantData::validate<bool>(DataType);
template bool ConstantData::validate<float>(DataType);

const std::vector<std::vector<std::string>>& ConstantData::values() const
{
    return _values;
}

DataType ConstantData::data_type() const
{
    return _data_type;
}

} // namespace ckw
