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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERCASTTEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERCASTTEST_H

#include "ckw/TileInfo.h"
#include "ckw/types/ConvertPolicy.h"
#include "ckw/types/DataType.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/Common.h"
#include "validation/tests/common/KernelWriterInterceptor.h"

#include <cstdint>
#include <vector>

namespace ckw
{

class CLKernelWriterCastTest : public ITest
{
public:
    CLKernelWriterCastTest()
    {
        _tests.push_back({ 1, 1, DataType::Fp16, 1, 1, DataType::Fp32, ConvertPolicy::None, "G0__dst = convert_half(G0__src);\n" }); // Scalar.

        _tests.push_back({ 1, 3, DataType::Int32, 1, 3, DataType::Fp16, ConvertPolicy::Saturate, "G0__dst = convert_int3_sat(G0__src);\n" }); // Whole vector.

        _tests.push_back({ 2, 4, DataType::Uint16, 2, 4, DataType::Int8, ConvertPolicy::Saturate, "G0__dst__0 = convert_ushort4_sat(G0__src__0);\nG0__dst__1 = convert_ushort4_sat(G0__src__1);\n" }); // Whole tile.

        _tests.push_back({ 2, 3, DataType::Int8, 1, 3, DataType::Uint8, ConvertPolicy::None, "G0__dst__0 = convert_char3(G0__src);\nG0__dst__1 = convert_char3(G0__src);\n" }); // Y-dimension broadcast.

        _tests.push_back({ 2, 4, DataType::Fp16, 2, 1, DataType::Fp32, ConvertPolicy::None, "G0__dst__0 = (half4)convert_half(G0__src__0);\nG0__dst__1 = (half4)convert_half(G0__src__1);\n" }); // X-dimension broadcast.

        _tests.push_back({ 2, 3, DataType::Fp32, 1, 1, DataType::Fp16, ConvertPolicy::None, "G0__dst__0 = (float3)convert_float(G0__src);\nG0__dst__1 = (float3)convert_float(G0__src);\n" }); // X and y dimension broadcast.
    }

    bool run() override
    {
        int32_t test_no          = 0;
        bool    all_tests_passed = true;

        for(const auto &test : _tests)
        {
            KernelWriterInterceptor<CLKernelWriter> writer;

            auto dst = writer.declare_tile("dst", TileInfo(test.dst_data_type, test.dst_height, test.dst_width));
            auto src = writer.declare_tile("src", TileInfo(test.src_data_type, test.src_height, test.src_width));

            writer.start_capture_code();

            writer.op_cast(dst, src, test.policy);

            VALIDATE_TEST(writer.check_added_code(test.expected_code), all_tests_passed, test_no++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterCastTest";
    }

private:
    struct TestInfo
    {
        int32_t       dst_height;
        int32_t       dst_width;
        DataType      dst_data_type;
        int32_t       src_height;
        int32_t       src_width;
        DataType      src_data_type;
        ConvertPolicy policy;
        std::string   expected_code;
    };

    std::vector<TestInfo> _tests{};
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERCASTTEST_H
