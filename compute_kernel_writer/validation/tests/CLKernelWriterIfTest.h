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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERIFTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERIFTEST_H

#include "ckw/Error.h"
#include "ckw/TileInfo.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/Common.h"
#include "validation/tests/common/KernelWriterInterceptor.h"

#include <cstdint>

namespace ckw
{

class CLKernelWriterIfTest : public ITest
{
public:
    CLKernelWriterIfTest()
    {
    }

    bool run() override
    {
        int32_t test_no          = 0;
        bool    all_tests_passed = true;

        KernelWriterInterceptor<CLKernelWriter> writer;

        auto lhs = writer.declare_tile("lhs", TileInfo(DataType::Fp32, 1, 1));
        auto rhs = writer.declare_tile("rhs", TileInfo(DataType::Fp32, 1, 1));

        // The first if block.
        {
            writer.start_capture_code();

            writer.op_if(
                lhs, BinaryOp::Equal, rhs,
                [&]()
                {
                    auto tile = writer.declare_tile("tile", TileInfo(DataType::Fp16, 2, 3));
                    CKW_UNUSED(tile);
                });

            constexpr auto expected_code =
                "if (G0__lhs == G0__rhs)\n"
                "{\n"
                "half3 G1__tile__0;\n"
                "half3 G1__tile__1;\n"
                "}\n";

            VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_no++);
        }

        // The second if block - The ID space inside the if block should change.
        {
            writer.start_capture_code();

            writer.op_if(
                lhs, BinaryOp::Equal, rhs,
                [&]()
                {
                    auto tile = writer.declare_tile("tile", TileInfo(DataType::Fp16, 2, 3));
                    CKW_UNUSED(tile);
                });

            constexpr auto expected_code =
                "if (G0__lhs == G0__rhs)\n"
                "{\n"
                "half3 G2__tile__0;\n"
                "half3 G2__tile__1;\n"
                "}\n";

            VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_no++);
        }

        // The if-else block - The ID space in each block should change.
        {
            writer.start_capture_code();

            writer.op_if(
                lhs, BinaryOp::Equal, rhs,
                [&]()
                {
                    auto tile = writer.declare_tile("tile", TileInfo(DataType::Fp16, 2, 3));
                    CKW_UNUSED(tile);
                });
            writer.op_else(
                [&]()
                {
                    auto tile = writer.declare_tile("tile", TileInfo(DataType::Uint8, 1, 4));
                    CKW_UNUSED(tile);
                });

            constexpr auto expected_code =
                "if (G0__lhs == G0__rhs)\n"
                "{\n"
                "half3 G3__tile__0;\n"
                "half3 G3__tile__1;\n"
                "}\n"
                "else\n"
                "{\n"
                "uchar4 G4__tile;\n"
                "}\n";

            VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_no++);
        }

        // If-else if block.
        {
            writer.start_capture_code();

            writer.op_if(
                lhs, BinaryOp::Equal, rhs,
                [&]()
                {
                    auto tile = writer.declare_tile("tile", TileInfo(DataType::Fp32, 1, 3));
                    CKW_UNUSED(tile);
                });
            writer.op_else_if(
                lhs, BinaryOp::Less, rhs,
                [&]()
                {
                    auto tile = writer.declare_tile("tile", TileInfo(DataType::Int8, 1, 4));
                    CKW_UNUSED(tile);
                });

            constexpr auto expected_code =
                "if (G0__lhs == G0__rhs)\n"
                "{\n"
                "float3 G5__tile;\n"
                "}\n"
                "else if (G0__lhs < G0__rhs)\n"
                "{\n"
                "char4 G6__tile;\n"
                "}\n";

            VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_no++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterIfTest";
    }
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERIFTEST_H
