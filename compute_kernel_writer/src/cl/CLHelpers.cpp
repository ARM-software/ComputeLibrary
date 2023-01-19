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
#include "ckw/Error.h"
#include "ckw/Types.h"

#include "src/cl/CLHelpers.h"

namespace ckw
{
bool cl_validate_vector_length(int32_t len)
{
    bool valid_vector_length = true;
    if(len < 1 || len > 16 || (len > 4 && len < 8) || (len > 8 && len < 16))
    {
        valid_vector_length = false;
    }
    return valid_vector_length;
}

std::string cl_get_variable_datatype_as_string(DataType dt, int32_t len)
{
    if(cl_validate_vector_length(len) == false)
    {
        COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported vector length");
        return "";
    }

    std::string res;
    switch(dt)
    {
        case DataType::Fp32:
            res += "float";
            break;
        case DataType::Fp16:
            res += "half";
            break;
        case DataType::Int8:
            res += "char";
            break;
        case DataType::Uint8:
            res += "uchar";
            break;
        case DataType::Uint16:
            res += "ushort";
            break;
        case DataType::Int16:
            res += "short";
            break;
        case DataType::Uint32:
            res += "uint";
            break;
        case DataType::Int32:
            res += "int";
            break;
        case DataType::Bool:
            res += "bool";
            break;
        default:
            COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported datatype");
            return "";
    }

    if(len > 1)
    {
        res += std::to_string(len);
    }

    return res;
}
} // namespace ckw