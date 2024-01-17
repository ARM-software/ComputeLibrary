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

#include "src/cl/CLKernelWriter.h"

#include "ckw/Error.h"
#include "ckw/Kernel.h"
#include "ckw/TensorSampler.h"
#include "ckw/TileOperand.h"
#include "ckw/types/DataType.h"
#include "ckw/types/MemoryOperation.h"
#include "ckw/types/TargetLanguage.h"

#include "src/cl/CLHelpers.h"
#include "src/cl/CLTensorArgument.h"
#include "src/cl/CLTile.h"
#include "src/cl/helpers/CLMemoryOpBufferHelper.h"
#include "src/cl/helpers/CLMemoryOpImage2dHelper.h"
#include "src/cl/helpers/ICLMemoryOpHelper.h"
#include "src/ITensorComponent.h"
#include "src/TileView.h"
#include "src/types/DataTypeHelpers.h"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

namespace
{
std::string generate_cl_extensions()
{
    std::string ext = R"(
#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif // defined(cl_khr_fp16)

#if defined(cl_arm_printf)
#pragma OPENCL EXTENSION cl_arm_printf : enable
#endif // defined(cl_arm_printf);

#define inf (INFINITY)
)";
    return ext;
}
} // namespace

namespace ckw
{

CLKernelWriter::CLKernelWriter()  = default;
CLKernelWriter::~CLKernelWriter() = default;

std::unique_ptr<Kernel> CLKernelWriter::emit_kernel(const std::string &name)
{
    std::string code;
    code += generate_cl_extensions();
    code += "__kernel void ";
    code += name;
    code += "\n(\n";

    // Create the list of arguments.
    std::vector<KernelArgument> arguments;

    for (const auto &tensor : _tensors)
    {
        const auto tensor_id = tensor->info().id();

        const auto storages   = tensor->storages();
        const auto components = tensor->components();

        for (const auto &storage : storages)
        {
            code += cl_get_variable_storagetype_as_string(storage.type);
            code += " ";
            code += storage.val;
            code += ",\n";

            arguments.emplace_back(tensor_id, storage.type);
        }

        for (const auto &component : components)
        {
            const auto &tile      = component->tile();
            const auto &tile_info = tile.info();

            CKW_ASSERT(tile.is_scalar());

            code += cl_get_variable_datatype_as_string(tile_info.data_type(), 1);
            code += " ";
            code += tile.name();
            code += ",\n";

            arguments.emplace_back(tensor_id, component->component_type());
        }
    }

    if (code.size() >= 2 && code[code.size() - 2] == ',' && code[code.size() - 1] == '\n')
    {
        // Remove the last comma in the argument list.
        code.pop_back();
        code[code.size() - 1] = '\n';
    }

    code += ")\n{\n";

    code += _body_source_code;

    code += "}\n";

    return std::make_unique<Kernel>(TargetLanguage::OpenCL, arguments, code);
}

void CLKernelWriter::op_assign(const TileOperand &dst, const TileOperand &src)
{
    const auto dst_view = to_cl_tile_view(dst);
    const auto src_view = to_cl_tile_view(src);

    const auto dst_w = dst_view.width();
    const auto dst_h = dst_view.height();
    const auto src_w = src_view.width();

    const auto data_type_str = cl_get_variable_datatype_as_string(dst_view.data_type(), dst_w);

    const auto        broadcast_src_x = dst_w != 1 && src_w == 1;
    const std::string src_prefix      = broadcast_src_x ? "(" + data_type_str + ")" : "";

    CKW_ASSERT_MSG(src_view.data_type() == dst_view.data_type(), "Source and destination type must match.");
    CKW_ASSERT_MSG(src_view.height() == dst_h || src_view.height() == 1,
                   "Tile height must match or source is broadcasting in y dimension.");
    CKW_ASSERT_MSG(src_w == dst_w || src_w == 1, "Tile width must match or source is broadcasting in x dimension.");

    // Broadcasting on y dimension is automatic (see CLTile::vector).
    for (int32_t y = 0; y < dst_h; ++y)
    {
        append_code(dst_view.vector(y).str, " = ", src_prefix, src_view.vector(y).str, ";\n");
    }
}

void CLKernelWriter::op_cast(const TileOperand &dst, const TileOperand &src, ConvertPolicy policy)
{
    const auto dst_view = to_cl_tile_view(dst);
    const auto src_view = to_cl_tile_view(src);

    const auto dst_w = dst_view.width();
    const auto dst_h = dst_view.height();
    const auto src_w = src_view.width();

    const auto dst_type = dst_view.data_type();

    const auto convert_type_str = cl_get_variable_datatype_as_string(dst_type, src_w);
    const auto dst_type_str     = cl_get_variable_datatype_as_string(dst_type, dst_w);

    const std::string sat = policy == ConvertPolicy::Saturate ? "_sat" : "";

    CKW_ASSERT_IF(policy == ConvertPolicy::Saturate, !is_data_type_float(dst_type));

    const auto        broadcast_x = dst_w != 1 && src_w == 1;
    const std::string prefix      = broadcast_x ? "(" + dst_type_str + ")" : "";

    CKW_ASSERT_MSG(src_view.height() == dst_h || src_view.height() == 1,
                   "Tile height must match or source is broadcasting in y dimension.");
    CKW_ASSERT_MSG(src_w == dst_w || src_w == 1, "Tile width must match or source is broadcasting in x dimension.");

    // Broadcasting on y dimension is automatic (see CLTile::vector).
    if (src_view.data_type() == dst_view.data_type())
    {
        for (int32_t y = 0; y < dst_h; ++y)
        {
            append_code(dst_view.vector(y).str, " = ", src_view.vector(y).str, ";\n");
        }
    }
    else
    {
        for (int32_t y = 0; y < dst_h; ++y)
        {
            append_code(dst_view.vector(y).str, " = ", prefix, "convert_", convert_type_str, sat, "(",
                        src_view.vector(y).str, ");\n");
        }
    }
}

void CLKernelWriter::op_unary(const TileOperand &dst, UnaryOp op, const TileOperand &src)
{
    const auto dst_view = to_cl_tile_view(dst);
    const auto src_view = to_cl_tile_view(src);

    const auto dst_w = dst_view.width();
    const auto dst_h = dst_view.height();
    const auto src_w = src_view.width();

    const auto data_type_str   = cl_get_variable_datatype_as_string(dst_view.data_type(), dst_w);
    const auto broadcast_src_x = dst_w != 1 && src_w == 1;

    const std::string src_prefix = broadcast_src_x ? "(" + data_type_str + ")" : "";

    const auto  op_info    = cl_get_unary_op(op);
    const auto  op_is_func = std::get<0>(op_info);
    const auto &op_name    = std::get<1>(op_info);
    const auto  op_prefix  = op_is_func ? op_name + "(" : op_name;
    const auto  op_suffix  = op_is_func ? ")" : "";

    CKW_ASSERT_MSG(src_view.data_type() == dst_view.data_type(), "Source and destination type must match.");
    CKW_ASSERT_MSG(src_view.height() == dst_h || src_view.height() == 1,
                   "Tile height must match or source is broadcasting in y dimension.");
    CKW_ASSERT_MSG(src_w == dst_w || src_w == 1, "Tile width must match or source is broadcasting in x dimension.");

    // Broadcasting on y dimension is automatic (see CLTile::vector).
    for (int32_t y = 0; y < dst_h; ++y)
    {
        append_code(dst_view.vector(y).str, " = ", src_prefix, op_prefix, src_view.vector(y).str, op_suffix, ";\n");
    }
}

void CLKernelWriter::op_binary(const TileOperand &dst, BinaryOp op, const TileOperand &first, const TileOperand &second)
{
    const auto dst_view = to_cl_tile_view(dst);
    const auto lhs_view = to_cl_tile_view(first);
    const auto rhs_view = to_cl_tile_view(second);

    const auto dst_w = dst_view.width();
    const auto dst_h = dst_view.height();
    const auto lhs_w = lhs_view.width();
    const auto rhs_w = rhs_view.width();

    const auto data_type = lhs_view.data_type();

    CKW_ASSERT_MSG(lhs_view.data_type() == rhs_view.data_type(), "LHS and RHS type must match.");

    if (op == BinaryOp::MatMul_Nt_T)
    {
        CKW_ASSERT_MSG(lhs_view.height() == dst_h, "LHS tile height must match the DST tile height");
        CKW_ASSERT_MSG(rhs_view.height() == dst_w, "RHS tile height must match the DST tile width");
        CKW_ASSERT_MSG(lhs_view.width() == rhs_view.width(), "LHS tile width must match the LHS tile width");

        CKW_ASSERT(is_data_type_float(data_type));

        for (int32_t y = 0; y < dst_h; ++y)
        {
            for (int32_t x = 0; x < dst_w; ++x)
            {
                for (int32_t k = 0; k < lhs_w; ++k)
                {
                    append_code(dst_view.scalar(y, x).str, " = fma(", lhs_view.scalar(y, k).str, ", ",
                                rhs_view.scalar(x, k).str, ", ", dst_view.scalar(y, x).str, ");\n");
                }
            }
        }
    }
    else
    {
        CKW_ASSERT_MSG(lhs_view.height() == dst_h || lhs_view.height() == 1,
                       "LHS tile height must match or source is broadcasting in y dimension.");
        CKW_ASSERT_MSG(rhs_view.height() == dst_h || rhs_view.height() == 1,
                       "RHS tile height must match or source is broadcasting in y dimension.");

        CKW_ASSERT_MSG(lhs_w == dst_w || lhs_w == 1,
                       "LHS tile width must match destination or LHS is broadcasting in x dimension.");
        CKW_ASSERT_MSG(rhs_w == dst_w || rhs_w == 1,
                       "RHS tile width must match destination or RHS is broadcasting in x dimension.");

        const auto  op_info    = cl_get_binary_op(op, data_type);
        const auto  op_is_func = std::get<0>(op_info);
        const auto &op_name    = std::get<1>(op_info);

        const auto data_type_str = cl_get_variable_datatype_as_string(data_type, dst_w);

        const auto broadcast_lhs_x = dst_w != 1 && lhs_w == 1;
        const auto broadcast_rhs_x = dst_w != 1 && rhs_w == 1;

        const std::string lhs_prefix = broadcast_lhs_x ? "(" + data_type_str + ")" : "";
        const std::string rhs_prefix = broadcast_rhs_x ? "(" + data_type_str + ")" : "";

        const std::string op_prefix    = op_is_func ? " = " + op_name + "(" : " = ";
        const std::string op_separator = op_is_func ? ", " : " " + op_name + " ";
        const std::string op_suffix    = op_is_func ? ");\n" : ";\n";

        // Broadcasting on y dimension is automatic (see CLTile::vector).
        for (int32_t y = 0; y < dst_h; ++y)
        {
            append_code(dst_view.vector(y).str, op_prefix, lhs_prefix, lhs_view.vector(y).str, op_separator, rhs_prefix,
                        rhs_view.vector(y).str, op_suffix);
        }
    }
}

void CLKernelWriter::op_ternary(
    const TileOperand &dst, TernaryOp op, const TileOperand &first, const TileOperand &second, const TileOperand &third)
{
    const auto dst_view    = to_cl_tile_view(dst);
    const auto first_view  = to_cl_tile_view(first);
    const auto second_view = to_cl_tile_view(second);
    const auto third_view  = to_cl_tile_view(third);

    const auto dst_w    = dst_view.width();
    const auto dst_h    = dst_view.height();
    const auto first_w  = first_view.width();
    const auto second_w = second_view.width();
    const auto third_w  = third_view.width();

    const auto data_type     = dst_view.data_type();
    const auto data_type_str = cl_get_variable_datatype_as_string(data_type, dst_w);

    const auto  op_info    = cl_get_ternary_op(op);
    const auto  op_is_func = std::get<0>(op_info);
    const auto &op_name    = std::get<1>(op_info);

    const auto broadcast_first_x  = dst_w != 1 && first_w == 1;
    const auto broadcast_second_x = dst_w != 1 && second_w == 1;
    const auto broadcast_third_x  = dst_w != 1 && third_w == 1;

    const std::string first_prefix  = broadcast_first_x ? "(" + data_type_str + ")" : "";
    const std::string second_prefix = broadcast_second_x ? "(" + data_type_str + ")" : "";
    const std::string third_prefix  = broadcast_third_x ? "(" + data_type_str + ")" : "";

    CKW_ASSERT_MSG(op_is_func, "The only supported ternary operator is function.");
    CKW_ASSERT_MSG(second_view.data_type() == dst_view.data_type(), "2nd source and destination type must match.");
    CKW_ASSERT_MSG(third_view.data_type() == dst_view.data_type(), "3rd source and destination type must match.");

    CKW_ASSERT_MSG(first_view.height() == dst_h || first_view.height() == 1,
                   "1st tile height must match or source is broadcasting in y dimension.");
    CKW_ASSERT_MSG(second_view.height() == dst_h || second_view.height() == 1,
                   "2nd tile height must match or source is broadcasting in y dimension.");
    CKW_ASSERT_MSG(third_view.height() == dst_h || third_view.height() == 1,
                   "3rd tile height must match or source is broadcasting in y dimension.");

    CKW_ASSERT_MSG(first_w == dst_w || first_w == 1,
                   "1st tile width must match or source is broadcasting in x dimension.");
    CKW_ASSERT_MSG(second_w == dst_w || second_w == 1,
                   "2nd tile width must match or source is broadcasting in x dimension.");
    CKW_ASSERT_MSG(third_w == dst_w || third_w == 1,
                   "3rd tile width must match or source is broadcasting in x dimension.");

    // Broadcasting on y dimension is automatic (see CLTile::vector).
    for (int32_t y = 0; y < dst_h; ++y)
    {
        append_code(dst_view.vector(y).str, " = ", op_name, "(", first_prefix, first_view.vector(y).str, ", ",
                    second_prefix, second_view.vector(y).str, ", ", third_prefix, third_view.vector(y).str, ");\n");
    }
}

void CLKernelWriter::op_if_generic(
    const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body, bool is_else_if)
{
    const auto lhs_view = to_cl_tile_view(lhs);
    const auto rhs_view = to_cl_tile_view(rhs);

    const auto op_name = std::get<1>(cl_get_binary_op(op, lhs_view.data_type()));
    CKW_ASSERT(op == BinaryOp::Less || op == BinaryOp::LessEqual || op == BinaryOp::Equal ||
               op == BinaryOp::GreaterEqual || op == BinaryOp::Greater);

    CKW_ASSERT(lhs_view.is_scalar());
    CKW_ASSERT(rhs_view.is_scalar());

    if (is_else_if)
    {
        append_code("else ");
    }

    append_code("if (", lhs_view.scalar(0, 0).str, " ", op_name, " ", rhs_view.scalar(0, 0).str, ")\n{\n");
    write_body(body);
    append_code("}\n");
}

void CLKernelWriter::op_if(const TileOperand           &lhs,
                           BinaryOp                     op,
                           const TileOperand           &rhs,
                           const std::function<void()> &body)
{
    op_if_generic(lhs, op, rhs, body, false /* is_else_if */);
}

void CLKernelWriter::op_else_if(const TileOperand           &lhs,
                                BinaryOp                     op,
                                const TileOperand           &rhs,
                                const std::function<void()> &body)
{
    op_if_generic(lhs, op, rhs, body, true /* is_else_if */);
}

void CLKernelWriter::op_else(const std::function<void()> &body)
{
    append_code("else\n{\n");
    write_body(body);
    append_code("}\n");
}

void CLKernelWriter::op_for_loop(const TileOperand           &var,
                                 BinaryOp                     cond_op,
                                 const TileOperand           &cond_value,
                                 const TileOperand           &update_var,
                                 AssignmentOp                 update_op,
                                 const TileOperand           &update_value,
                                 const std::function<void()> &body)
{
    const auto var_view          = to_cl_tile_view(var);
    const auto cond_value_view   = to_cl_tile_view(cond_value);
    const auto update_var_view   = to_cl_tile_view(update_var);
    const auto update_value_view = to_cl_tile_view(update_value);

    CKW_ASSERT(var_view.is_scalar());
    CKW_ASSERT(cond_value_view.is_scalar());
    CKW_ASSERT(update_var_view.is_scalar());
    CKW_ASSERT(update_value_view.is_scalar());

    CKW_ASSERT(var_view.data_type() == cond_value_view.data_type());
    CKW_ASSERT(update_var_view.data_type() == update_value_view.data_type());

    const auto cond_op_name = std::get<1>(cl_get_binary_op(cond_op, var_view.data_type()));
    CKW_ASSERT(cond_op == BinaryOp::Less || cond_op == BinaryOp::LessEqual || cond_op == BinaryOp::Equal ||
               cond_op == BinaryOp::GreaterEqual || cond_op == BinaryOp::Greater);

    append_code("for (; ", var_view.scalar(0, 0).str, " ", cond_op_name, " ", cond_value_view.scalar(0, 0).str, "; ",
                update_var_view.scalar(0, 0).str, " ", cl_get_assignment_op_as_string(update_op), " ",
                update_value_view.scalar(0, 0).str, ")\n{\n");
    write_body(body);
    append_code("}\n");
}

void CLKernelWriter::op_return()
{
    append_code("return;\n");
}

void CLKernelWriter::op_get_global_id(const TileOperand &dst, int32_t dim)
{
    const auto tile_view = to_cl_tile_view(dst);

    CKW_ASSERT(tile_view.is_scalar());
    CKW_ASSERT(tile_view.data_type() == DataType::Int32 || tile_view.data_type() == DataType::Uint32);

    CKW_ASSERT(dim >= 0 && dim <= 2);

    append_code(tile_view.scalar(0, 0).str, " = get_global_id(", std::to_string(dim), ");\n");
}

void CLKernelWriter::op_print(const std::string &prefix, const std::vector<TileOperand> &operands)
{
    std::string format_code;
    std::string args_code;

    for (auto &op : operands)
    {
        const auto tile_view = to_cl_tile_view(op);

        const auto name      = tile_view.name();
        const auto width     = tile_view.width();
        const auto height    = tile_view.height();
        const auto data_type = tile_view.data_type();

        // Construct the format specifier to print out one row of the tile.
        std::string row_format("%");

        if (width > 1)
        {
            row_format += "v" + std::to_string(width);
        }

        switch (data_type)
        {
            case DataType::Fp32:
                row_format += "hlg";
                break;
            case DataType::Fp16:
                row_format += "hg";
                break;
            case DataType::Int32:
            case DataType::Bool:
                row_format += (width > 1) ? "hli" : "i";
                break;
            case DataType::Int16:
                row_format += "hi";
                break;
            case DataType::Int8:
                row_format += "hhi";
                break;
            case DataType::Uint32:
                row_format += (width > 1) ? "hlu" : "u";
                break;
            case DataType::Uint16:
                row_format += "hu";
                break;
            case DataType::Uint8:
                row_format += "hhu";
                break;
            default:
                CKW_THROW_MSG("Unsupported data type!");
        }

        if (width > 1)
        {
            row_format = "[" + row_format + "]";
        }

        // Construct the format specifier for the printf statement.
        format_code += name + " = ";

        if (height == 1)
        {
            format_code += row_format;
        }
        else
        {
            format_code += "[" + row_format;
            for (int32_t row = 1; row < height; ++row)
            {
                format_code += ", " + row_format;
            }
            format_code += "]";
        }

        format_code += "\\n";

        // Construct the variable arguments for the printf statement.
        for (int32_t row = 0; row < height; ++row)
        {
            args_code += ", " + tile_view.vector(row).str;
        }
    }

    append_code("printf(\"", prefix, "\\n", format_code, "\"", args_code, ");\n");
}

void CLKernelWriter::op_comment(const std::string &text)
{
#ifdef COMPUTE_KERNEL_WRITER_DEBUG_ENABLED

    CKW_ASSERT(text.find("\n") == text.npos);
    CKW_ASSERT(text.find("\r") == text.npos);

    append_code("// ", text, "\n");

#else // COMPUTE_KERNEL_WRITER_DEBUG_ENABLED

    CKW_UNUSED(text);

#endif // COMPUTE_KERNEL_WRITER_DEBUG_ENABLED
}

const std::string &CLKernelWriter::body_source_code() const
{
    return _body_source_code;
}

TensorOperand CLKernelWriter::declare_tensor_argument(const std::string &name, const TensorInfo &info)
{
    const auto fullname = generate_full_name(name);

    auto       tensor  = std::make_unique<CLTensorArgument>(fullname, info, false /* return_dims_by_value */);
    const auto operand = create_tensor_operand(*tensor);

    _tensors.insert(std::move(tensor));

    return operand;
}

TileOperand CLKernelWriter::declare_tile(const std::string &name, const TileInfo &tile_info)
{
    const std::string fullname = generate_full_name(name);

    const int32_t  height    = tile_info.height();
    const int32_t  width     = tile_info.width();
    const DataType data_type = tile_info.data_type();

    CKW_ASSERT_MSG(std::find_if(_tiles.begin(), _tiles.end(),
                                [=](const std::unique_ptr<CLTile> &e)
                                { return e->name() == fullname; }) == _tiles.end(),
                   "There is already a tile with name: " + fullname);

    auto tile = std::make_unique<CLTile>(fullname, tile_info);

    for (int32_t row = 0; row < height; ++row)
    {
        const std::string cl_type = cl_get_variable_datatype_as_string(data_type, width);
        append_code(cl_type, " ", tile->vector(row).str, ";\n");
    }

    const auto operand = create_tile_operand(*tile);

    _tiles.insert(std::move(tile));

    return operand;
}

TileOperand CLKernelWriter::declare_constant_tile(const ConstantData &data)
{
    auto              tile    = std::make_unique<CLTile>(get_values(data), get_data_type(data));
    const TileOperand operand = create_tile_operand(*tile);
    _constant_tiles.insert(std::move(tile));

    return operand;
}

void CLKernelWriter::op_write_raw_code(const std::string &raw_code)
{
    append_code(raw_code);
}

TileView<CLTile> CLKernelWriter::to_cl_tile_view(const TileOperand &operand) const
{
    const auto     tile_and_area = get_tile(operand);
    ITile         &tile          = std::get<0>(tile_and_area);
    const TileArea area          = std::get<1>(tile_and_area);

#ifdef COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED
    // Check if the tile is a CLTile created by this kernel writer.

    {
        bool found = false;

        for (const auto &t : _tiles)
        {
            if (&tile == t.get())
            {
                found = true;
                break;
            }
        }

        for (const auto &t : _constant_tiles)
        {
            if (&tile == t.get())
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            for (const auto &t : _tensors)
            {
                const auto components = t->components();

                for (const auto component : components)
                {
                    if (&tile == &component->tile())
                    {
                        found = true;
                        break;
                    }
                }

                if (found)
                {
                    break;
                }
            }
        }

        CKW_ASSERT_MSG(found, "The tile is not found!");
    }
#endif // COMPUTE_KERNEL_WRITER_ASSERTS_ENABLED

    return {static_cast<CLTile &>(tile), area};
}

void CLKernelWriter::op_load(const TileOperand   &tile_op,
                             const TensorOperand &tensor_op,
                             TensorSampler       &sampler,
                             const TileOperand   &x,
                             const TileOperand   &y,
                             const TileOperand   &z,
                             const TileOperand   &batch)
{
    const CLTile dilation_x({{"1"}}, DataType::Int32);
    const CLTile dilation_y({{"1"}}, DataType::Int32);

    op_load_store(MemoryOperation::Load, tile_op, tensor_op, sampler, x, y, z, batch, dilation_x, dilation_y,
                  false /* indirect buffer */);
}

void CLKernelWriter::op_load_dilated(const TileOperand   &tile_op,
                                     const TensorOperand &tensor_op,
                                     TensorSampler       &sampler,
                                     const TileOperand   &x,
                                     const TileOperand   &y,
                                     const TileOperand   &z,
                                     const TileOperand   &batch,
                                     const TileOperand   &dilation_x,
                                     const TileOperand   &dilation_y)
{
    const auto dil_x_view = to_cl_tile_view(dilation_x);
    const auto dil_y_view = to_cl_tile_view(dilation_y);

    op_load_store(MemoryOperation::Load, tile_op, tensor_op, sampler, x, y, z, batch, dil_x_view, dil_y_view,
                  false /* indirect buffer */);
}

void CLKernelWriter::op_store(const TensorOperand &tensor_op,
                              const TileOperand   &tile_op,
                              TensorSampler       &sampler,
                              const TileOperand   &x,
                              const TileOperand   &y,
                              const TileOperand   &z,
                              const TileOperand   &batch)
{
    const CLTile dilation_x({{"1"}}, DataType::Int32);
    const CLTile dilation_y({{"1"}}, DataType::Int32);

    op_load_store(MemoryOperation::Store, tile_op, tensor_op, sampler, x, y, z, batch, dilation_x, dilation_y,
                  false /* indirect buffer */);
}

void CLKernelWriter::op_store_dilated(const TensorOperand &tensor_op,
                                      const TileOperand   &tile_op,
                                      TensorSampler       &sampler,
                                      const TileOperand   &x,
                                      const TileOperand   &y,
                                      const TileOperand   &z,
                                      const TileOperand   &batch,
                                      const TileOperand   &dilation_x,
                                      const TileOperand   &dilation_y)
{
    const auto dil_x_view = to_cl_tile_view(dilation_x);
    const auto dil_y_view = to_cl_tile_view(dilation_y);

    op_load_store(MemoryOperation::Store, tile_op, tensor_op, sampler, x, y, z, batch, dil_x_view, dil_y_view,
                  false /* indirect buffer */);
}

void CLKernelWriter::op_load_indirect(const TileOperand   &tile_op,
                                      const TensorOperand &tensor_op,
                                      TensorSampler       &sampler,
                                      const TileOperand   &x,
                                      const TileOperand   &y,
                                      const TileOperand   &z,
                                      const TileOperand   &batch)
{
    const CLTile dilation_x({{"1"}}, DataType::Int32);
    const CLTile dilation_y({{"1"}}, DataType::Int32);

    op_load_store(MemoryOperation::Load, tile_op, tensor_op, sampler, x, y, z, batch, dilation_x, dilation_y,
                  true /* indirect buffer */);
}

void CLKernelWriter::op_load_store(MemoryOperation         op,
                                   const TileOperand      &tile_op,
                                   const TensorOperand    &tensor_op,
                                   TensorSampler          &sampler,
                                   const TileOperand      &x,
                                   const TileOperand      &y,
                                   const TileOperand      &z,
                                   const TileOperand      &batch,
                                   const TileView<CLTile> &dilation_x,
                                   const TileView<CLTile> &dilation_y,
                                   bool                    indirect_buffer)
{
    CKW_UNUSED(dilation_x);
    CKW_ASSERT(dilation_x.is_scalar());
    CKW_ASSERT(dilation_y.is_scalar());
    CKW_ASSERT(dilation_x.scalar(0, 0).str == "((int)(1))"); // Dilation in x dimension is not implemented yet

    if (indirect_buffer)
    {
        CKW_ASSERT(dilation_y.scalar(0, 0).str == "((int)(1))" && dilation_x.scalar(0, 0).str == "((int)(1))");
    }

    ITensor &tensor = get_tensor(tensor_op);

    const auto tile       = to_cl_tile_view(tile_op);
    const auto x_tile     = to_cl_tile_view(x).full_tile();
    const auto y_tile     = to_cl_tile_view(y).full_tile();
    const auto z_tile     = to_cl_tile_view(z).full_tile();
    const auto batch_tile = to_cl_tile_view(batch).full_tile();

    std::unique_ptr<ICLMemoryOpHelper> helper;
    switch (sampler.storage())
    {
        case TensorStorageType::BufferUint8Ptr:
            helper = std::make_unique<CLMemoryOpBufferHelper>(this, &tensor, &sampler, op, tile);
            break;
        case TensorStorageType::Texture2dReadOnly:
        case TensorStorageType::Texture2dWriteOnly:
            helper = std::make_unique<CLMemoryOpImage2dHelper>(this, &tensor, &sampler, op, tile);
            break;
        default:
            CKW_THROW_MSG("Unsupported tensor storage");
    }

    CKW_ASSERT(x_tile.is_scalar());
    CKW_ASSERT(z_tile.is_scalar());
    CKW_ASSERT_IF(indirect_buffer, y_tile.info().width() == 1);
    CKW_ASSERT_IF(!indirect_buffer, y_tile.is_scalar());
    CKW_ASSERT(batch_tile.is_scalar());

    helper->initialize(&x_tile, &z_tile, &batch_tile);

    for (int row = 0; row < tile.height(); ++row)
    {
        if (!indirect_buffer)
        {
            std::string coord_y = y_tile.scalar(0, 0).str + " + " + std::to_string(row);

            if (dilation_y.scalar(0, 0).str != "((int)(1))")
            {
                coord_y += " * " + dilation_y.scalar(0, 0).str;
            }

            helper->write_row(row, coord_y);
        }
        else
        {
            helper->write_row(row, y_tile.scalar(row, 0).str);
        }
    }

    helper->finalize();
}

} // namespace ckw
