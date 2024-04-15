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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERUNARYEXPRESSIONTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERUNARYEXPRESSIONTEST_H

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

class CLKernelWriterUnaryExpressionTest : public ITest
{
public:
    CLKernelWriterUnaryExpressionTest()
    {
        // dst_height, dst_width, src_height, src_width, data_type, op, expected_code

        _tests.push_back({ 1, 1, 1, 1, DataType::Uint32, UnaryOp::BitwiseNot, "G0__dst = ~G0__src;\n" }); // Scalar.

        _tests.push_back({ 1, 3, 1, 3, DataType::Int16, UnaryOp::LogicalNot, "G0__dst = !G0__src;\n" }); // Whole vector.

        _tests.push_back({ 2, 4, 2, 4, DataType::Int8, UnaryOp::Exp, "G0__dst__0 = exp(G0__src__0);\nG0__dst__1 = exp(G0__src__1);\n" }); // Whole tile.

        _tests.push_back({ 2, 3, 1, 3, DataType::Uint8, UnaryOp::Log, "G0__dst__0 = log(G0__src);\nG0__dst__1 = log(G0__src);\n" }); // Y-dimension broadcast.

        _tests.push_back({ 2, 4, 2, 1, DataType::Uint16, UnaryOp::Sqrt, "G0__dst__0 = (ushort4)sqrt(G0__src__0);\nG0__dst__1 = (ushort4)sqrt(G0__src__1);\n" }); // X-dimension broadcast.

        _tests.push_back({ 2, 3, 1, 1, DataType::Int32, UnaryOp::Round, "G0__dst__0 = (int3)round(G0__src);\nG0__dst__1 = (int3)round(G0__src);\n" }); // X and y dimension broadcast.
    }

    bool run() override
    {
        int32_t test_no          = 0;
        bool    all_tests_passed = true;

        for(const auto &test : _tests)
        {
            KernelWriterInterceptor<CLKernelWriter> writer;

            auto dst = writer.declare_tile("dst", TileInfo(test.data_type, test.dst_height, test.dst_width));
            auto src = writer.declare_tile("src", TileInfo(test.data_type, test.src_height, test.src_width));

            writer.start_capture_code();

            writer.op_unary(dst, test.op, src);

            VALIDATE_TEST(writer.check_added_code(test.expected_code), all_tests_passed, test_no++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterUnaryExpressionTest";
    }

private:
    struct TestInfo
    {
        int32_t     dst_height;
        int32_t     dst_width;
        int32_t     src_height;
        int32_t     src_width;
        DataType    data_type;
        UnaryOp     op;
        std::string expected_code;
    };

    std::vector<TestInfo> _tests{};
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERUNARYEXPRESSIONTEST_H
