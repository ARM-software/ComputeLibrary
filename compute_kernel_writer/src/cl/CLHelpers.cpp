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

#include "src/cl/CLHelpers.h"

#include "ckw/Error.h"
#include "ckw/types/DataType.h"
#include "ckw/types/Operators.h"
#include "ckw/types/TensorStorageType.h"

#include "src/types/DataTypeHelpers.h"

namespace ckw
{
bool cl_validate_vector_length(int32_t len)
{
    bool valid_vector_length = true;
    if (len < 1 || len > 16 || (len > 4 && len < 8) || (len > 8 && len < 16))
    {
        valid_vector_length = false;
    }
    return valid_vector_length;
}

std::string cl_get_variable_datatype_as_string(DataType dt, int32_t len)
{
    if (cl_validate_vector_length(len) == false)
    {
        CKW_THROW_MSG("Unsupported vector length");
        return "";
    }

    std::string res;
    switch (dt)
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
            CKW_THROW_MSG("Unsupported datatype");
            return "";
    }

    if (len > 1)
    {
        res += std::to_string(len);
    }

    return res;
}

int32_t cl_round_up_to_nearest_valid_vector_width(int32_t width)
{
    switch (width)
    {
        case 1:
            return 1;
        case 2:
            return 2;
        case 3:
            return 3;
        case 4:
            return 4;
        case 5:
        case 6:
        case 7:
        case 8:
            return 8;
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
            return 16;
        default:
            CKW_THROW_MSG("Unsupported width to convert to OpenCL vector");
            return 0;
    }
}

std::string cl_get_variable_storagetype_as_string(TensorStorageType storage)
{
    std::string res;
    switch (storage)
    {
        case TensorStorageType::BufferUint8Ptr:
            res += "__global uchar*";
            break;
        case TensorStorageType::Texture2dReadOnly:
            res += "__read_only image2d_t";
            break;
        case TensorStorageType::Texture2dWriteOnly:
            res += "__write_only image2d_t";
            break;
        default:
            CKW_THROW_MSG("Unsupported storage type");
    }

    return res;
}

std::string cl_get_assignment_op_as_string(AssignmentOp op)
{
    switch (op)
    {
        case AssignmentOp::Increment:
            return "+=";

        case AssignmentOp::Decrement:
            return "-=";

        default:
            CKW_THROW_MSG("Unsupported assignment operator!");
    }
}

std::tuple<bool, std::string> cl_get_unary_op(UnaryOp op)
{
    switch (op)
    {
        case UnaryOp::LogicalNot:
            return {false, "!"};

        case UnaryOp::BitwiseNot:
            return {false, "~"};

        case UnaryOp::Exp:
            return {true, "exp"};

        case UnaryOp::Tanh:
            return {true, "tanh"};

        case UnaryOp::Sqrt:
            return {true, "sqrt"};

        case UnaryOp::Erf:
            return {true, "erf"};

        case UnaryOp::Fabs:
            return {true, "fabs"};

        case UnaryOp::Log:
            return {true, "log"};

        case UnaryOp::Round:
            return {true, "round"};

        case UnaryOp::Floor:
            return {true, "floor"};

        default:
            CKW_THROW_MSG("Unsupported unary operation!");
    }
}

std::tuple<bool, std::string> cl_get_binary_op(BinaryOp op, DataType data_type)
{
    const auto is_float = is_data_type_float(data_type);

    switch (op)
    {
        case BinaryOp::Add:
            return {false, "+"};

        case BinaryOp::Sub:
            return {false, "-"};

        case BinaryOp::Mul:
            return {false, "*"};

        case BinaryOp::Div:
            return {false, "/"};

        case BinaryOp::Mod:
            return {false, "%"};

        case BinaryOp::Equal:
            return {false, "=="};

        case BinaryOp::Less:
            return {false, "<"};

        case BinaryOp::LessEqual:
            return {false, "<="};

        case BinaryOp::Greater:
            return {false, ">"};

        case BinaryOp::GreaterEqual:
            return {false, ">="};

        case BinaryOp::LogicalAnd:
            return {false, "&&"};

        case BinaryOp::LogicalOr:
            return {false, "||"};

        case BinaryOp::BitwiseXOR:
            return {false, "^"};

        case BinaryOp::Min:
            return {true, is_float ? "fmin" : "min"};

        case BinaryOp::Max:
            return {true, is_float ? "fmax" : "max"};

        default:
            CKW_THROW_MSG("Unsupported binary operator/function!");
    }
}

std::tuple<bool, std::string> cl_get_ternary_op(TernaryOp op)
{
    switch (op)
    {
        case TernaryOp::Select:
            return {true, "select"};

        case TernaryOp::Clamp:
            return {true, "clamp"};

        default:
            CKW_THROW_MSG("Unsupported ternary function!");
    }
}

std::string cl_data_type_rounded_up_to_valid_vector_width(DataType dt, int32_t width)
{
    std::string   data_type;
    const int32_t w = cl_round_up_to_nearest_valid_vector_width(width);
    data_type += cl_get_variable_datatype_as_string(dt, 1);
    if (w != 1)
    {
        data_type += std::to_string(w);
    }
    return data_type;
}

std::vector<int32_t> cl_decompose_vector_width(int32_t vector_width)
{
    std::vector<int32_t> x;

    switch (vector_width)
    {
        case 0:
            break;
        case 1:
        case 2:
        case 3:
        case 4:
        case 8:
        case 16:
            x.push_back(vector_width);
            break;
        case 5:
            x.push_back(4);
            x.push_back(1);
            break;
        case 6:
            x.push_back(4);
            x.push_back(2);
            break;
        case 7:
            x.push_back(4);
            x.push_back(3);
            break;
        case 9:
            x.push_back(8);
            x.push_back(1);
            break;
        case 10:
            x.push_back(8);
            x.push_back(2);
            break;
        case 11:
            x.push_back(8);
            x.push_back(3);
            break;
        case 12:
            x.push_back(8);
            x.push_back(4);
            break;
        case 13:
            x.push_back(8);
            x.push_back(4);
            x.push_back(1);
            break;
        case 14:
            x.push_back(8);
            x.push_back(4);
            x.push_back(2);
            break;
        case 15:
            x.push_back(8);
            x.push_back(4);
            x.push_back(3);
            break;

        default:
            CKW_THROW_MSG("Vector width is too large");
    }
    return x;
}

} // namespace ckw
