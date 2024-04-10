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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERTERNARYOPTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERTERNARYOPTEST_H

#include "ckw/TileInfo.h"
#include "ckw/types/DataType.h"
#include "ckw/types/Operators.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/Common.h"
#include "validation/tests/common/KernelWriterInterceptor.h"

#include <cstdint>
#include <vector>

namespace ckw
{

class CLKernelWriterTernaryOpTest : public ITest
{
public:
    CLKernelWriterTernaryOpTest()
    {
        // dst_height, dst_width, op0_height, op0_width, op1_height, op1_width, op2_height, op2_width, data_type, op, expected_code

        _tests.push_back({ 1, 1, 1, 1, 1, 1, 1, 1, DataType::Fp32, TernaryOp::Select, "G0__dst = select(G0__op0, G0__op1, G0__op2);\n" }); // Scalar.

        _tests.push_back({ 1, 3, 1, 3, 1, 3, 1, 3, DataType::Fp16, TernaryOp::Clamp, "G0__dst = clamp(G0__op0, G0__op1, G0__op2);\n" }); // Whole vector.

        _tests.push_back({ 2, 4, 2, 4, 2, 4, 2, 4, DataType::Int8, TernaryOp::Select, "G0__dst__0 = select(G0__op0__0, G0__op1__0, G0__op2__0);\nG0__dst__1 = select(G0__op0__1, G0__op1__1, G0__op2__1);\n" }); // Whole tile.

        _tests.push_back({ 2, 3, 1, 3, 2, 3, 2, 3, DataType::Uint8, TernaryOp::Clamp, "G0__dst__0 = clamp(G0__op0, G0__op1__0, G0__op2__0);\nG0__dst__1 = clamp(G0__op0, G0__op1__1, G0__op2__1);\n" }); // 1st operand y-dimension broadcast.

        _tests.push_back({ 2, 3, 2, 3, 2, 1, 2, 3, DataType::Fp32, TernaryOp::Select, "G0__dst__0 = select(G0__op0__0, (float3)G0__op1__0, G0__op2__0);\nG0__dst__1 = select(G0__op0__1, (float3)G0__op1__1, G0__op2__1);\n" }); // 2nd operand x-dimension broadcast.

        _tests.push_back({ 2, 3, 1, 3, 2, 1, 1, 1, DataType::Fp16, TernaryOp::Clamp, "G0__dst__0 = clamp(G0__op0, (half3)G0__op1__0, (half3)G0__op2);\nG0__dst__1 = clamp(G0__op0, (half3)G0__op1__1, (half3)G0__op2);\n" }); // 1st operand y-, 2nd operand x-, 3rd operand x- and y-dimension broadcast.
    }

    bool run() override
    {
        int32_t test_no          = 0;
        bool    all_tests_passed = true;

        for(const auto &test : _tests)
        {
            KernelWriterInterceptor<CLKernelWriter> writer;

            auto dst = writer.declare_tile("dst", TileInfo(test.data_type, test.dst_height, test.dst_width));
            auto op0 = writer.declare_tile("op0", TileInfo(DataType::Bool, test.op0_height, test.op0_width));
            auto op1 = writer.declare_tile("op1", TileInfo(test.data_type, test.op1_height, test.op1_width));
            auto op2 = writer.declare_tile("op2", TileInfo(test.data_type, test.op2_height, test.op2_width));

            writer.start_capture_code();

            writer.op_ternary(dst, test.op, op0, op1, op2);

            VALIDATE_TEST(writer.check_added_code(test.expected_code), all_tests_passed, test_no++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterTernaryOpTest";
    }

private:
    struct TestInfo
    {
        int32_t     dst_height;
        int32_t     dst_width;
        int32_t     op0_height;
        int32_t     op0_width;
        int32_t     op1_height;
        int32_t     op1_width;
        int32_t     op2_height;
        int32_t     op2_width;
        DataType    data_type;
        TernaryOp   op;
        std::string expected_code;
    };

    std::vector<TestInfo> _tests{};
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERTERNARYOPTEST_H
