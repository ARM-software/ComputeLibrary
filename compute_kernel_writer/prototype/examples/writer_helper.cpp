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
#include "../include/ckw/KernelWriterHelper.h"
#include "ckw/TensorTileSampler.h"

#include <iostream>

using namespace ckw;

TensorTileSampler create_simple_sampler(KernelWriter& writer)
{
    TensorTileSampler sampler;

    constexpr int32_t m0 = 1;
    constexpr int32_t n0 = 1;

    auto &gid_0 = writer.declare_tile("gid_0", DataType::Int32);
    auto &gid_1 = writer.declare_tile("gid_1", DataType::Int32);
    auto &gid_2 = writer.declare_tile("gid_2", DataType::Int32);

    auto &const_0 = writer.declare_tile("0", 0);

    writer.op_get_global_id(gid_0, 0);
    writer.op_get_global_id(gid_1, 1);
    writer.op_get_global_id(gid_2, 2);

    sampler.x(gid_0);
    sampler.y(gid_1);
    sampler.z(gid_2);
    sampler.b(const_0);

    sampler.width(n0);
    sampler.height(m0);

    sampler.format(TensorSamplerFormat::C_WH_1);
    sampler.address_mode_x(TensorSamplerAddressModeX::None);
    sampler.address_mode_y(TensorSamplerAddressModeY::ClampToBorder);
    sampler.address_mode_z(TensorSamplerAddressModeZ::Skip);

    return sampler;
}

int main()
{
    Kernel kernel("test", GpuTargetLanguage::OpenCL);
    KernelWriterHelper<KernelWriter> writer(kernel);

    const TensorInfo src_info(DataType::Fp32, TensorShape({ 1, 1, 1, 1, 1 }), TensorDataLayout::Nhwc, 0);
    const TensorInfo dst_info(DataType::Fp32, TensorShape({ 1, 1, 1, 1, 1 }), TensorDataLayout::Nhwc, 1);

    auto &src_tensor = writer.declare_tensor_argument("src", src_info);
    auto &dst_tensor = writer.declare_tensor_argument("dst", dst_info);

    const auto sampler = create_simple_sampler(writer);

    auto &src = writer.declare_tile("src_tile", TileInfo(src_tensor.data_type(), sampler.height(), sampler.width()));
    auto &other = writer.declare_tile("other_tile", TileInfo(src_tensor.data_type(), sampler.height(), sampler.width()));
    auto &dst = writer.declare_tile("dst_tile", TileInfo(src_tensor.data_type(), sampler.height(), sampler.width()));

    writer.op_load(src, src_tensor, sampler);
    writer.op_load(other, src_tensor, sampler);
    writer.op_load(dst, dst_tensor, sampler);

    auto test = dst ^ src ^ other;
    auto other_test = logical_and(dst, src, other);
    writer.op_assign(dst, logical_and(dst, src, other));
    writer.op_assign(dst, test);
    writer.op_assign(dst, other_test);
    writer.op_assign(dst, operator^(operator^(dst, src), other));

    writer.op_if(exp(src) == dst, [&]{
        writer.op_binary_expression(dst, src, BinaryOp::Add, src);
    }).op_else_if(exp(src) > dst, [&]{
        writer.op_binary_expression(dst, src, BinaryOp::Add, src);
    }).op_else([&] {
        writer.op_assign(dst, src);
    });

    writer.op_assign(dst, src + src * src);
    writer.op_assign(dst, src * max(src, dst) + src);
    writer.op_assign(dst, src * select(src, dst, src) + src);

    writer.op_assign(dst, src ^ dst);
    writer.op_assign(dst, ~src);

    writer.op_for_loop(dst < src, dst += src, [&]{
        writer.op_assign(dst, src + dst);
    });

    writer.op_assign(dst += src);
    writer.op_assign(dst += exp(src));

    std::cout << "======== KERNEL ========" << std::endl;
    std::cout << writer.generate_code() << std::endl;
}