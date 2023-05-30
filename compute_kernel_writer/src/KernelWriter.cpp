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

TensorOperand &KernelWriter::create_tensor_argument(const char *name, const TensorInfo &info)
{
    const auto var_name = generate_variable_name(name);

    _impl->declare_argument(var_name, create_impl_tensor_info(info));

    auto operand = new TensorOperand(var_name, info);
    register_operand(operand, false);

    return *operand;
}

TileOperand &KernelWriter::create_tile_argument(const char *name, int32_t value)
{
    const auto var_name = generate_variable_name(name);

    auto operand = new TileOperand(var_name, value);
    register_operand(operand, false);

    return *operand;
}

std::string KernelWriter::generate_variable_name(const char *name) const
{
    std::stringstream var_name;

    var_name << "_" << _id_space << "_" << name;

    return var_name.str();
}

void KernelWriter::register_operand(OperandBase *operand, bool declaring)
{
    const auto &name     = operand->name();
    auto       &operands = _kernel->operands();

    CKW_ASSERT(operands.find(name) == operands.end());
    operands[name] = std::unique_ptr<OperandBase>(operand);

    if(declaring && !operand->is_constant())
    {
        const auto tile = reinterpret_cast<TileOperand *>(operand);

        const auto &info = tile->tile_info();
        _impl->declare_tile(tile->name(), prototype::TileInfo(info.data_type(), info.width(), info.height()));
    }
}

// =================================================================================================
// Load and store
// =================================================================================================

void KernelWriter::op_load(TileOperand &tile, TensorOperand &tensor, const TensorTileSampler &sampler)
{
    auto impl_tensor = prototype::TensorOperand(
        tensor.name(),
        prototype::GpuSampler{
            sampler.format(),
            prototype::GpuSamplerTensorStorage::BufferUint8Ptr,
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
    auto impl_tensor = prototype::TensorOperand(
        tensor.name(),
        prototype::GpuSampler{
            sampler.format(),
            prototype::GpuSamplerTensorStorage::BufferUint8Ptr,
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

void KernelWriter::op_assign(TileOperand &dst, const TileOperand &src)
{
    auto impl_dst = dst.create_impl_operand(_impl.get());
    auto impl_src = src.create_impl_operand(_impl.get());

    _impl->op_assign(impl_dst, impl_src);
}

void KernelWriter::op_binary_expression(TileOperand &dst, const TileOperand &lhs, const TileOperand &rhs, BinaryOp op)
{
    auto impl_lhs = lhs.create_impl_operand(_impl.get());
    auto impl_rhs = rhs.create_impl_operand(_impl.get());
    auto impl_dst = dst.create_impl_operand(_impl.get());

    _impl->op_binary_expression(impl_dst, impl_lhs, op, impl_rhs);
}

void KernelWriter::op_scalar_function(TileOperand &dst, const TileOperand &src, ScalarUnaryFunction opcode)
{
    auto impl_dst = dst.create_impl_operand(_impl.get());
    auto impl_src = src.create_impl_operand(_impl.get());

    _impl->op_scalar_function(impl_dst, impl_src, opcode);
}

// =================================================================================================
// Misc
// =================================================================================================

void KernelWriter::op_get_global_id(TileOperand &dst, int32_t dim)
{
    _impl->op_get_global_id(prototype::Operand(dst.name()), dim);
}

// =================================================================================================
// Code generation
// =================================================================================================

std::string KernelWriter::generate_code()
{
    return prototype::generate_code(*_kernel->impl(), _kernel->name());
}

} // namespace ckw
