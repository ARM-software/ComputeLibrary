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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERPRINT_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERPRINT_H

#include "ckw/TileInfo.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/Common.h"
#include "validation/tests/common/KernelWriterInterceptor.h"

namespace ckw
{

class CLKernelWriterPrintTest : public ITest
{
public:
    CLKernelWriterPrintTest()
    {
    }

    bool run() override
    {
        bool all_tests_passed = true;

        KernelWriterInterceptor<CLKernelWriter> writer;

        const auto tile2x3f16 = writer.declare_tile("tile2x3f16", TileInfo(DataType::Fp16, 2, 3));
        const auto tile1x2i32 = writer.declare_tile("tile1x2i32", TileInfo(DataType::Int32, 1, 2));
        const auto tile2x1s32 = writer.declare_tile("tile2x1s32", TileInfo(DataType::Int32, 2, 1));
        const auto tile1x1u32 = writer.declare_tile("tile1x1u32", TileInfo(DataType::Uint32, 1, 1));

        writer.start_capture_code();

        writer.op_print("debug_log", { tile2x3f16, tile1x2i32, tile2x1s32, tile1x1u32 });

        constexpr auto expected_code =
            "printf(\"debug_log\\nG0__tile2x3f16 = [[%v3hg], [%v3hg]]\\nG0__tile1x2i32 = [%v2hli]\\nG0__tile2x1s32 = [%i, %i]\\nG0__tile1x1u32 = %u\\n\", "
            "G0__tile2x3f16__0, G0__tile2x3f16__1, G0__tile1x2i32, G0__tile2x1s32__0, G0__tile2x1s32__1, G0__tile1x1u32);\n";

        VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, 0);

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterPrintTest";
    }
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERPRINT_H
