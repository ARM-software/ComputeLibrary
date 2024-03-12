/*
 * Copyright (c) 2016-2023 Arm Limited.
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

#include "arm_compute/core/utils/DataTypeUtils.h"

#include <map>

namespace arm_compute
{
const std::string &string_from_data_type(DataType dt)
{
    static std::map<DataType, const std::string> dt_map = {
        {DataType::UNKNOWN, "UNKNOWN"},
        {DataType::S8, "S8"},
        {DataType::U8, "U8"},
        {DataType::S16, "S16"},
        {DataType::U16, "U16"},
        {DataType::S32, "S32"},
        {DataType::U32, "U32"},
        {DataType::S64, "S64"},
        {DataType::U64, "U64"},
        {DataType::F16, "F16"},
        {DataType::F32, "F32"},
        {DataType::F64, "F64"},
        {DataType::SIZET, "SIZET"},
        {DataType::QSYMM8, "QSYMM8"},
        {DataType::QSYMM8_PER_CHANNEL, "QSYMM8_PER_CHANNEL"},
        {DataType::QASYMM8, "QASYMM8"},
        {DataType::QASYMM8_SIGNED, "QASYMM8_SIGNED"},
        {DataType::QSYMM16, "QSYMM16"},
        {DataType::QASYMM16, "QASYMM16"},
    };

    return dt_map[dt];
}

DataType data_type_from_name(const std::string &name)
{
    static const std::map<std::string, DataType> data_types = {
        {"f16", DataType::F16},
        {"f32", DataType::F32},
        {"qasymm8", DataType::QASYMM8},
        {"qasymm8_signed", DataType::QASYMM8_SIGNED},
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return data_types.at(utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch (const std::out_of_range &)
    {
        ARM_COMPUTE_ERROR_VAR("Invalid data type name: %s", name.c_str());
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}

} // namespace arm_compute
