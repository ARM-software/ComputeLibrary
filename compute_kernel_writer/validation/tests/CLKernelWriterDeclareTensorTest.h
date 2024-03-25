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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERDECLARETENSORTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERDECLARETENSORTEST_H

#include "ckw/Error.h"
#include "ckw/Kernel.h"
#include "ckw/KernelArgument.h"
#include "ckw/TensorInfo.h"
#include "ckw/types/TensorComponentType.h"
#include "ckw/types/TensorDataLayout.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/Common.h"

namespace ckw
{

class CLKernelWriterDeclareTensorTest : public ITest
{
public:
    CLKernelWriterDeclareTensorTest()
    {
    }

    std::string name() override
    {
        return "CLKernelWriterDeclareTensorTest";
    }

    bool run() override
    {
        auto all_tests_passed = true;

        CLKernelWriter writer;

        auto src = writer.declare_tensor_argument("src", TensorInfo(DataType::Fp32, TensorShape{ 2, 3, 4, 5 }, TensorDataLayout::Nhwc, 0));
        auto dst = writer.declare_tensor_argument("dst", TensorInfo(DataType::Fp32, TensorShape{ 6, 7, 8, 9 }, TensorDataLayout::Nhwc, 1));

        auto src_dim0           = src.dim0();
        auto src_stride2        = src.stride2();
        auto src_offset_element = src.offset_first_element_in_bytes();

        auto dst_dim1 = dst.dim0();

        auto src_dim0_again = src.dim0();

        CKW_UNUSED(src_dim0, src_stride2, src_offset_element, dst_dim1, src_dim0_again);

        const auto kernel = writer.emit_kernel("test_kernel");

        const std::string expected_code =
            "__kernel void test_kernel\n"
            "(\n"
            "int G0__src_dim0,\n"
            "int G0__src_stride2,\n"
            "int G0__src_offset_first_element,\n"
            "int G0__dst_dim0\n"
            ")\n"
            "{\n"
            "}\n";

        const auto &actual_code = kernel->source_code();

        int test_id = 0;
        VALIDATE_TEST(kernel->arguments().size() == 4, all_tests_passed, test_id++);
        test_tensor_component_argument(kernel->arguments()[0], 0, TensorComponentType::Dim0, all_tests_passed, test_id);
        test_tensor_component_argument(kernel->arguments()[1], 0, TensorComponentType::Stride2, all_tests_passed, test_id);
        test_tensor_component_argument(kernel->arguments()[2], 0, TensorComponentType::OffsetFirstElement, all_tests_passed, test_id);
        test_tensor_component_argument(kernel->arguments()[3], 1, TensorComponentType::Dim0, all_tests_passed, test_id);
        VALIDATE_TEST(actual_code == expected_code, all_tests_passed, test_id++);

        return all_tests_passed;
    }

    void test_tensor_component_argument(const KernelArgument &arg, int32_t tensor_id, TensorComponentType component_type, bool &all_tests_passed, int &test_id)
    {
        VALIDATE_TEST(arg.type() == KernelArgument::Type::TensorComponent, all_tests_passed, test_id++);
        VALIDATE_TEST(arg.id() == tensor_id, all_tests_passed, test_id++);
        VALIDATE_TEST(arg.tensor_component_type() == component_type, all_tests_passed, test_id++);
    }
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERDECLARETENSORTEST_H
