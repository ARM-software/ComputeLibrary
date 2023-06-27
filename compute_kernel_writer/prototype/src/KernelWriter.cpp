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

#include "ckw/KernelWriter.h"
#include "ckw/Error.h"
#include "ckw/TensorInfo.h"
#include "ckw/TensorOperand.h"
#include "src/Prototype.h"

#include <sstream>

namespace ckw
{

namespace
{

inline prototype::TensorInfo create_impl_tensor_info(const TensorInfo &info)
{
    return prototype::TensorInfo{ info.shape(), info.data_type(), info.data_layout(), info.id() };
}

} // namespace

// =================================================================================================
// Constructors and destructor
// =================================================================================================

KernelWriter::KernelWriter(Kernel &kernel)
    : _kernel(&kernel),
      _impl_attr(std::make_unique<prototype::GpuKernelWriterAttribute>()),
      _impl(prototype::GpuKernelWriterFactory::create(_impl_attr.get(), kernel.impl()))
{
    _impl->set_IdSpace(1);
}

KernelWriter::~KernelWriter()
{
}

// =================================================================================================
// Scope management
// =================================================================================================

int32_t KernelWriter::id_space() const
{
    return _id_space;
}

KernelWriter &KernelWriter::id_space(int32_t id_space)
{
    CKW_ASSERT(id_space <= _max_id_space);

    _id_space = id_space;
    return *this;
}

int32_t KernelWriter::next_id_space()
{
    id_space(++_max_id_space);
    return _id_space;
}

// =================================================================================================
// Tensor and tile declaration
// =================================================================================================

TensorOperand &KernelWriter::declare_tensor_argument(const std::string &name, const TensorInfo &info, TensorStorageType storage_type)
{
    const auto var_name = generate_variable_name(name);

    _impl->declare_argument(var_name, create_impl_tensor_info(info));

    auto &operand = _kernel->register_operand(std::make_unique<TensorOperand>(var_name, info, storage_type));

    return operand;
}

TileOperand &KernelWriter::declare_tile_argument(const std::string &name, int32_t value)
{
    const auto var_name = generate_variable_name(name);

    auto &operand = _kernel->register_operand(std::make_unique<TileOperand>(var_name, value));

    return operand;
}

std::string KernelWriter::generate_variable_name(const std::string &name) const
{
    std::stringstream var_name;

    var_name << "_" << _id_space << "_" << name;

    return var_name.str();
}

TileOperand &KernelWriter::declare_tile_operand(std::unique_ptr<TileOperand> operand_ptr)
{
    auto       &operand = _kernel->register_operand(std::move(operand_ptr));
    const auto &name    = operand.name();

    if(!operand.is_constant())
    {
        const auto &info = operand.tile_info();

        _impl->declare_tile(
            name,
            prototype::TileInfo(info.data_type(), info.width(), info.height()));
    }

    return operand;
}

// =================================================================================================
// Load and store
// =================================================================================================

void KernelWriter::op_load(TileOperand &tile, TensorOperand &tensor, const TensorTileSampler &sampler)
{
    prototype::TensorOperand impl_tensor(
        tensor.name(),
        prototype::GpuSampler{
            sampler.format(),
            prototype::to_gpu_tensor_storage(tensor.storage_type()),
            sampler.address_mode_x(),
            sampler.address_mode_y(),
            sampler.address_mode_z() });

    auto impl_x = sampler.x().create_impl_operand(_impl.get());
    auto impl_y = sampler.y().create_impl_operand(_impl.get());
    auto impl_z = sampler.z().create_impl_operand(_impl.get());
    auto impl_b = sampler.b().create_impl_operand(_impl.get());

    auto impl_dst = tile.create_impl_operand(_impl.get());

    _impl->op_load_immediate(impl_tensor, impl_dst, impl_x, impl_y, impl_z, impl_b);
}

void KernelWriter::op_store(TensorOperand &tensor, const TileOperand &tile, const TensorTileSampler &sampler)
{
    prototype::TensorOperand impl_tensor(
        tensor.name(),
        prototype::GpuSampler{
            sampler.format(),
            prototype::to_gpu_tensor_storage(tensor.storage_type()),
            sampler.address_mode_x(),
            sampler.address_mode_y(),
            sampler.address_mode_z() });
    auto impl_src = tile.create_impl_operand(_impl.get());
    auto impl_x   = sampler.x().create_impl_operand(_impl.get());
    auto impl_y   = sampler.y().create_impl_operand(_impl.get());
    auto impl_z   = sampler.z().create_impl_operand(_impl.get());
    auto impl_b   = sampler.b().create_impl_operand(_impl.get());

    _impl->op_store_immediate(impl_tensor, impl_src, impl_x, impl_y, impl_z, impl_b);
}

// =================================================================================================
// Data processing
// =================================================================================================

void KernelWriter::op_assign(const TileOperand &dst, const TileOperand &src)
{
    auto impl_dst = dst.create_impl_operand(_impl.get());
    auto impl_src = src.create_impl_operand(_impl.get());

    _impl->op_assign(impl_dst, impl_src);
}

void KernelWriter::op_cast_expression(const TileOperand &dst, const TileOperand &src, const ConvertPolicy policy)
{
    auto impl_dst = dst.create_impl_operand(_impl.get());
    auto impl_src = src.create_impl_operand(_impl.get());

    _impl->op_cast_expression(impl_dst, impl_src, policy);
}

void KernelWriter::op_binary_expression(const TileOperand &dst, const TileOperand &lhs, BinaryOp op, const TileOperand &rhs)
{
    auto impl_lhs = lhs.create_impl_operand(_impl.get());
    auto impl_rhs = rhs.create_impl_operand(_impl.get());
    auto impl_dst = dst.create_impl_operand(_impl.get());

    _impl->op_binary_expression(impl_dst, impl_lhs, op, impl_rhs);
}

void KernelWriter::op_unary_expression(const TileOperand &dst, UnaryOp op, const TileOperand &src)
{
    auto impl_dst = dst.create_impl_operand(_impl.get());
    auto impl_src = src.create_impl_operand(_impl.get());

    _impl->op_unary_expression(impl_dst, op, impl_src);
}

void KernelWriter::op_unary_elementwise_function(const TileOperand &dst, UnaryFunction opcode, const TileOperand &src)
{
    auto impl_dst = dst.create_impl_operand(_impl.get());
    auto impl_src = src.create_impl_operand(_impl.get());

    _impl->op_unary_elementwise_function(impl_dst, opcode, impl_src);
}

void KernelWriter::op_binary_elementwise_function(const TileOperand &dst, BinaryFunction opcode, const TileOperand &first, const TileOperand &second)
{
    auto impl_dst    = dst.create_impl_operand(_impl.get());
    auto impl_first  = first.create_impl_operand(_impl.get());
    auto impl_second = second.create_impl_operand(_impl.get());

    _impl->op_binary_elementwise_function(impl_dst, opcode, impl_first, impl_second);
}

void KernelWriter::op_ternary_elementwise_function(const TileOperand &dst, TernaryFunction opcode, const TileOperand &first, const TileOperand &second, const TileOperand &third)
{
    auto impl_dst    = dst.create_impl_operand(_impl.get());
    auto impl_first  = first.create_impl_operand(_impl.get());
    auto impl_second = second.create_impl_operand(_impl.get());
    auto impl_third  = third.create_impl_operand(_impl.get());

    _impl->op_ternary_elementwise_function(impl_dst, opcode, impl_first, impl_second, impl_third);
}

void KernelWriter::op_if(const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body)
{
    auto impl_lhs = lhs.create_impl_operand(_impl.get());
    auto impl_rhs = rhs.create_impl_operand(_impl.get());

    _impl->op_if_header(impl_lhs, op, impl_rhs);
    _impl->compound_statement_begin();
    body();
    _impl->compound_statement_end();
}

void KernelWriter::op_else_if(const TileOperand &lhs, BinaryOp op, const TileOperand &rhs, const std::function<void()> &body)
{
    auto impl_lhs = lhs.create_impl_operand(_impl.get());
    auto impl_rhs = rhs.create_impl_operand(_impl.get());

    _impl->op_else_if_header(impl_lhs, op, impl_rhs);
    _impl->compound_statement_begin();
    body();
    _impl->compound_statement_end();
}

void KernelWriter::op_else(const std::function<void()> &body)
{
    _impl->op_else_header();
    _impl->compound_statement_begin();
    body();
    _impl->compound_statement_end();
}

void KernelWriter::op_for_loop(const TileOperand &var_name, BinaryOp cond_op, const TileOperand &cond_value_name, const TileOperand &update_var_name, AssignmentOp update_op, const TileOperand &update_value_name, const std::function<void()> &body)
{
    auto impl_var_name          = var_name.create_impl_operand(_impl.get());
    auto impl_cond_value_name   = cond_value_name.create_impl_operand(_impl.get());
    auto impl_update_var_name   = update_var_name.create_impl_operand(_impl.get());
    auto impl_update_value_name = update_value_name.create_impl_operand(_impl.get());

    _impl->op_for_loop_header(impl_var_name, cond_op, impl_cond_value_name, impl_update_var_name, update_op, impl_update_value_name);
    _impl->compound_statement_begin();
    body();
    _impl->compound_statement_end();
}

// =================================================================================================
// Misc
// =================================================================================================

void KernelWriter::op_get_global_id(TileOperand &dst, int32_t dim)
{
    _impl->op_get_global_id(prototype::Operand(dst.name()), dim);
}

void KernelWriter::op_return()
{
    _impl->op_return();
}

// =================================================================================================
// Code generation
// =================================================================================================

std::string KernelWriter::generate_code()
{
    return prototype::generate_code(*_kernel->impl(), _kernel->name());
}

} // namespace ckw
