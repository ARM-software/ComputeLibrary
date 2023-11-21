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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERGETGLOBALIDTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERGETGLOBALIDTEST_H

#include "ckw/TileInfo.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/Common.h"
#include "validation/tests/common/KernelWriterInterceptor.h"

namespace ckw
{

class CLKernelWriterGetGlobalIdTest : public ITest
{
public:
    CLKernelWriterGetGlobalIdTest()
    {
    }

    bool run() override
    {
        bool all_tests_passed = true;

        KernelWriterInterceptor<CLKernelWriter> writer;

        auto gid = writer.declare_tile("gid", TileInfo(DataType::Int32));

        writer.start_capture_code();

        writer.op_get_global_id(gid, 0);
        writer.op_get_global_id(gid, 1);
        writer.op_get_global_id(gid, 2);

        constexpr auto expected_code = "G0__gid = get_global_id(0);\nG0__gid = get_global_id(1);\nG0__gid = get_global_id(2);\n";

        VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, 0);

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterGetGlobalIdTest";
    }
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERGETGLOBALIDTEST_H
