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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERCOMMENTTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERCOMMENTTEST_H

#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/Common.h"
#include "validation/tests/common/KernelWriterInterceptor.h"

namespace ckw
{

class CLKernelWriterCommentTest : public ITest
{
public:
    CLKernelWriterCommentTest()
    {
    }

    bool run() override
    {
        bool all_tests_passed = true;

        KernelWriterInterceptor<CLKernelWriter> writer;

        writer.op_comment("previous code");

        writer.start_capture_code();

        writer.op_comment("code under test 0");
        writer.op_comment("code under test 1");

#ifdef COMPUTE_KERNEL_WRITER_DEBUG_ENABLED
        constexpr auto expected_code = "// code under test 0\n// code under test 1\n";
#else  // COMPUTE_KERNEL_WRITER_DEBUG_ENABLED
        constexpr auto expected_code = "";
#endif // COMPUTE_KERNEL_WRITER_DEBUG_ENABLED

        VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, 0);

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterCommentTest";
    }
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERCOMMENTTEST_H
