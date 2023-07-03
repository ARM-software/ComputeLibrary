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
#include "ckw/KernelWriter.h"
#include "ckw/TensorOperand.h"
#include "ckw/TensorTileSampler.h"
#include "ckw/TileOperand.h"
#include "ckw/Types.h"

#include "common/ExampleComponentArgument.h"
#include "common/ExampleKernelWriter.h"
#include "common/ExampleScopedKernelWriter.h"

#include <iostream>
#include <vector>

using namespace ckw;

TensorTileSampler create_simple_sampler(ExampleScopedKernelWriter writer)
{
    TensorTileSampler sampler;

    constexpr int32_t m0 = 4;
    constexpr int32_t n0 = 4;

    auto &gid_0 = writer->declare_tile("gid_0", DataType::Int32);
    auto &gid_1 = writer->declare_tile("gid_1", DataType::Int32);
    auto &gid_2 = writer->declare_tile("gid_2", DataType::Int32);

    auto &const_0 = writer->declare_tile("0", 0);

    writer->op_get_global_id(gid_0, 0);
    writer->op_get_global_id(gid_1, 1);
    writer->op_get_global_id(gid_2, 2);

    sampler.x(gid_0);
    sampler.y(gid_1);
    sampler.z(const_0);
    sampler.b(gid_2);

    sampler.width(n0);
    sampler.height(m0);

    sampler.format(TensorSamplerFormat::C_WH_1);
    sampler.address_mode_x(TensorSamplerAddressModeX::None);
    sampler.address_mode_y(TensorSamplerAddressModeY::ClampToBorder);
    sampler.address_mode_z(TensorSamplerAddressModeZ::Skip);

    return sampler;
}

void op_binary_elementwise(ExampleScopedKernelWriter writer, std::vector<ExampleComponentArgument *> operands)
{
    auto lhs = operands.at(0);
    auto rhs = operands.at(1);
    auto dst = operands.at(2);

    // Load the LHS and RHS tile and prepare the tensor sampler.
    if(!lhs->has_tile() && !rhs->has_tile())
    {
        const auto sampler = create_simple_sampler(writer);

        writer->op_load_once(lhs, sampler);
        writer->op_load_once(rhs, sampler);
    }
    else if(lhs->has_tile())
    {
        const auto &sampler = lhs->tile_sampler();
        writer->op_load_once(rhs, sampler);
    }
    else
    {
        const auto &sampler = rhs->tile_sampler();
        writer->op_load_once(lhs, sampler);
    }

    auto       &lhs_tile = lhs->tile();
    auto       &rhs_tile = rhs->tile();
    const auto &sampler  = lhs->tile_sampler();

    // Prepare the output tile.
    if(!dst->has_tile())
    {
        auto &tile = writer->declare_tile("dst_tile", lhs_tile.tile_info());
        dst->init_virtual_tensor(tile, sampler);
    }

    auto &dst_tile = dst->tile();

    // Perform the operation.
    writer->op_binary_expression(dst_tile, lhs_tile, rhs_tile, BinaryOp::Add);
}

void op_exp(ExampleScopedKernelWriter writer, std::vector<ExampleComponentArgument *> operands)
{
    auto src = operands.at(0);
    auto dst = operands.at(1);

    // Load the source tile and prepare the sampler.
    if(!src->has_tile())
    {
        const auto sampler = create_simple_sampler(writer);
        writer->op_load_once(src, sampler);
    }

    auto       &src_tile = src->tile();
    const auto &sampler  = src->tile_sampler();

    // Prepare the output tile.
    if(!dst->has_tile())
    {
        auto &tile = writer->declare_tile("dst_tile", src_tile.tile_info());
        dst->init_virtual_tensor(tile, sampler);
    }

    auto &dst_tile = dst->tile();

    // Perform the operation.
    writer->op_scalar_function(dst_tile, src_tile, ScalarUnaryFunction::Exp);
}

void op_store(ExampleScopedKernelWriter writer, std::vector<ExampleComponentArgument *> operands)
{
    auto src = operands.at(0);
    auto dst = operands.at(1);

    auto       &src_tile   = src->tile();
    const auto &sampler    = src->tile_sampler();
    auto       &dst_tensor = dst->tensor();

    writer->op_store(dst_tensor, src_tile, sampler);
}

int main()
{
    Kernel          kernel("example", GpuTargetLanguage::OpenCL);
    ExampleKernelWriter root_writer(kernel);

    ExampleScopedKernelWriter writer(&root_writer);

    const TensorInfo src0_info(DataType::Fp32, TensorShape({ 3, 10, 20, 1, 1 }), TensorDataLayout::Nhwc, 0);
    const TensorInfo src1_info(DataType::Fp32, TensorShape({ 3, 10, 20, 1, 1 }), TensorDataLayout::Nhwc, 1);
    const TensorInfo dst_info(DataType::Fp32, TensorShape({ 3, 10, 20, 1, 1 }), TensorDataLayout::Nhwc, 2);

    ExampleComponentArgument src0(writer->create_tensor_argument("src0", src0_info));
    ExampleComponentArgument src1(writer->create_tensor_argument("src1", src1_info));
    ExampleComponentArgument dst(writer->create_tensor_argument("dst", dst_info));

    ExampleComponentArgument ans;

    op_binary_elementwise(writer, { &src0, &src1, &ans });
    op_exp(writer, { &ans, &ans });
    op_store(writer, { &ans, &dst });

    const auto code = root_writer.generate_code();
    std::cout << code;

    return 0;
}
