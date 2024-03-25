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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERBINARYOPTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERBINARYOPTEST_H

#include "ckw/TileInfo.h"
#include "ckw/types/DataType.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/Common.h"
#include "validation/tests/common/KernelWriterInterceptor.h"

#include <cstdint>
#include <vector>

namespace ckw
{

class CLKernelWriterBinaryOpTest : public ITest
{
public:
    CLKernelWriterBinaryOpTest()
    {
        // dst_height, dst_width, dst_data_type, lhs_height, lhs_width, rhs_height, rhs_width, src_data_type, op, expected_code
        _tests.push_back({ 1, 1, DataType::Fp32, 1, 1, 1, 1, DataType::Fp32, BinaryOp::Add, "G0__dst = G0__lhs + G0__rhs;\n" }); // Scalar.

        _tests.push_back({ 1, 3, DataType::Bool, 1, 3, 1, 3, DataType::Fp16, BinaryOp::Equal, "G0__dst = G0__lhs == G0__rhs;\n" }); // Whole vector.

        _tests.push_back({ 2, 4, DataType::Int8, 2, 4, 2, 4, DataType::Int8, BinaryOp::Min, "G0__dst__0 = min(G0__lhs__0, G0__rhs__0);\nG0__dst__1 = min(G0__lhs__1, G0__rhs__1);\n" }); // Whole tile.

        _tests.push_back({ 2, 3, DataType::Uint8, 1, 3, 2, 3, DataType::Uint8, BinaryOp::BitwiseXOR, "G0__dst__0 = G0__lhs ^ G0__rhs__0;\nG0__dst__1 = G0__lhs ^ G0__rhs__1;\n" }); // LHS y-dimension broadcast.

        _tests.push_back({ 2, 3, DataType::Bool, 2, 3, 1, 3, DataType::Fp32, BinaryOp::Less, "G0__dst__0 = G0__lhs__0 < G0__rhs;\nG0__dst__1 = G0__lhs__1 < G0__rhs;\n" }); // RHS y-dimension broadcast.

        _tests.push_back({ 2, 3, DataType::Fp16, 1, 3, 1, 3, DataType::Fp16, BinaryOp::Max, "G0__dst__0 = fmax(G0__lhs, G0__rhs);\nG0__dst__1 = fmax(G0__lhs, G0__rhs);\n" }); // LHS and RHS y-dimension broadcast.

        _tests.push_back({ 2, 4, DataType::Fp32, 2, 1, 2, 4, DataType::Fp32, BinaryOp::Div, "G0__dst__0 = (float4)G0__lhs__0 / G0__rhs__0;\nG0__dst__1 = (float4)G0__lhs__1 / G0__rhs__1;\n" }); // LHS x-dimension broadcast.

        _tests.push_back({ 2, 4, DataType::Fp16, 2, 4, 2, 1, DataType::Fp16, BinaryOp::Mod, "G0__dst__0 = G0__lhs__0 % (half4)G0__rhs__0;\nG0__dst__1 = G0__lhs__1 % (half4)G0__rhs__1;\n" }); // RHS x-dimension broadcast.

        _tests.push_back({ 2, 4, DataType::Bool, 2, 1, 2, 1, DataType::Fp32, BinaryOp::GreaterEqual, "G0__dst__0 = (float4)G0__lhs__0 >= (float4)G0__rhs__0;\nG0__dst__1 = (float4)G0__lhs__1 >= (float4)G0__rhs__1;\n" }); // LHS and RHS x-dimension broadcast.

        _tests.push_back({ 2, 3, DataType::Fp32, 2, 3, 2, 3, DataType::Fp32, BinaryOp::MatMul_Nt_T,
                           "G0__dst__0.s0 = fma(G0__lhs__0.s0, G0__rhs__0.s0, G0__dst__0.s0);\n"
                           "G0__dst__0.s0 = fma(G0__lhs__1.s0, G0__rhs__1.s0, G0__dst__0.s0);\n"
                           "G0__dst__0.s0 = fma(G0__lhs__1.s0, G0__rhs__1.s0, G0__dst__0.s0);\n"
                           "G0__dst__1.s0 = fma(G0__lhs__0.s0, G0__rhs__0.s1, G0__dst__1.s0);\n"
                           "G0__dst__1.s0 = fma(G0__lhs__1.s0, G0__rhs__1.s1, G0__dst__1.s0);\n"
                           "G0__dst__1.s0 = fma(G0__lhs__1.s0, G0__rhs__1.s1, G0__dst__1.s0);\n"
                           "G0__dst__1.s0 = fma(G0__lhs__0.s0, G0__rhs__0.s2, G0__dst__1.s0);\n"
                           "G0__dst__1.s0 = fma(G0__lhs__1.s0, G0__rhs__1.s2, G0__dst__1.s0);\n"
                           "G0__dst__1.s0 = fma(G0__lhs__1.s0, G0__rhs__1.s2, G0__dst__1.s0);\n"
                           "G0__dst__0.s1 = fma(G0__lhs__0.s1, G0__rhs__0.s0, G0__dst__0.s1);\n"
                           "G0__dst__0.s1 = fma(G0__lhs__1.s1, G0__rhs__1.s0, G0__dst__0.s1);\n"
                           "G0__dst__0.s1 = fma(G0__lhs__1.s1, G0__rhs__1.s0, G0__dst__0.s1);\n"
                           "G0__dst__1.s1 = fma(G0__lhs__0.s1, G0__rhs__0.s1, G0__dst__1.s1);\n"
                           "G0__dst__1.s1 = fma(G0__lhs__1.s1, G0__rhs__1.s1, G0__dst__1.s1);\n"
                           "G0__dst__1.s1 = fma(G0__lhs__1.s1, G0__rhs__1.s1, G0__dst__1.s1);\n"
                           "G0__dst__1.s1 = fma(G0__lhs__0.s1, G0__rhs__0.s2, G0__dst__1.s1);\n"
                           "G0__dst__1.s1 = fma(G0__lhs__1.s1, G0__rhs__1.s2, G0__dst__1.s1);\n"
                           "G0__dst__1.s1 = fma(G0__lhs__1.s1, G0__rhs__1.s2, G0__dst__1.s1);\n" });
    }

    bool run() override
    {
        int32_t test_no          = 0;
        bool    all_tests_passed = true;

        for(const auto &test : _tests)
        {
            KernelWriterInterceptor<CLKernelWriter> writer;

            auto dst = writer.declare_tile("dst", TileInfo(test.dst_data_type, test.dst_height, test.dst_width));
            auto lhs = writer.declare_tile("lhs", TileInfo(test.src_data_type, test.lhs_height, test.lhs_width));
            auto rhs = writer.declare_tile("rhs", TileInfo(test.src_data_type, test.rhs_height, test.rhs_width));

            writer.start_capture_code();

            writer.op_binary(dst, test.op, lhs, rhs);

            VALIDATE_TEST(writer.check_added_code(test.expected_code), all_tests_passed, test_no++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterBinaryOpTest";
    }

private:
    struct TestInfo
    {
        int32_t     dst_height;
        int32_t     dst_width;
        DataType    dst_data_type;
        int32_t     lhs_height;
        int32_t     lhs_width;
        int32_t     rhs_height;
        int32_t     rhs_width;
        DataType    src_data_type;
        BinaryOp    op;
        std::string expected_code;
    };

    std::vector<TestInfo> _tests{};
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERBINARYOPTEST_H
