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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITERDECLARECONSTANTTILETEST_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITERDECLARECONSTANTTILETEST_H

#include "ckw/TileInfo.h"
#include "ckw/types/ConstantData.h"
#include "ckw/types/DataType.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/KernelWriterInterceptor.h"
#include "validation/tests/common/Common.h"

#include <string>
#include <tuple>
#include <vector>

namespace ckw
{
class CLKernelWriterDeclareConstantTileTest : public ITest
{
    using TestConfig = std::tuple<ConstantData, DataType, int32_t, int32_t, std::string>;
public:
    CLKernelWriterDeclareConstantTileTest()
    {
        _configs = {
            // ConstantData, DataType, Height, Width
            {ConstantData({{1}}, DataType::Int32), DataType::Int32, 1, 1,
                "G0__tile = ((int)(1));\n"},
            {ConstantData({{1U}}, DataType::Uint32), DataType::Uint32, 1, 1,
                "G0__tile = ((uint)(1));\n"},
            {ConstantData({{1, 2}}, DataType::Int8), DataType::Int8, 1, 2,
                "G0__tile = ((char2)(1, 2));\n"},
            {ConstantData({{1, -2}, {-3, 4}}, DataType::Int32), DataType::Int32, 2, 2,
                "G0__tile__0 = ((int2)(1, -2));\nG0__tile__1 = ((int2)(-3, 4));\n"},
            {ConstantData({{1.0f, -2.0f}}, DataType::Fp16), DataType::Fp16, 1, 2,
                "G0__tile = ((half2)(1.000000000e+00, -2.000000000e+00));\n"},
            {ConstantData({{/* FLT_MAX */ 340282346638528859811704183484516925440.0f, -2.0f, 3.0f}}, DataType::Fp32), DataType::Fp32, 1, 3,
                "G0__tile = ((float3)(3.402823466e+38, -2.000000000e+00, 3.000000000e+00));\n"},
            {ConstantData({{1.0f, -1e-20f, 2e-20f, /* FLT_EPS */ 1.1920928955078125e-7f}}, DataType::Fp32), DataType::Fp32, 1, 4,
                "G0__tile = ((float4)(1.000000000e+00, -9.999999683e-21, 1.999999937e-20, 1.192092896e-07));\n"},
            {ConstantData({{0.5f, 2.1e-30f, /* FLT_MIN */ 1.175494350822287507969e-38f}}, DataType::Fp32), DataType::Fp32, 1, 3,
                "G0__tile = ((float3)(5.000000000e-01, 2.099999969e-30, 1.175494351e-38));\n"},
            {ConstantData({{true}, {false}, {false}}, DataType::Bool), DataType::Bool, 3, 1,
                "G0__tile__0 = ((bool)(1));\nG0__tile__1 = ((bool)(0));\nG0__tile__2 = ((bool)(0));\n"}
        };
    }

    bool run() override
    {
        bool all_tests_passed = true;
        int test_idx = 0;

        for(TestConfig _config: _configs)
        {
            KernelWriterInterceptor<CLKernelWriter> writer;
            const ConstantData const_data = std::get<0>(_config);
            const DataType data_type = std::get<1>(_config);
            const size_t height = std::get<2>(_config);
            const size_t width = std::get<3>(_config);
            const std::string expected_code = std::get<4>(_config);

            TileOperand tile = writer.declare_tile("tile", TileInfo(data_type, height, width));
            writer.start_capture_code();
            TileOperand const_tile = writer.declare_constant_tile(const_data);
            writer.op_assign(tile, const_tile);

            VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_idx++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterDeclareConstantTileTest";
    }

private:
    std::vector<TestConfig> _configs {};
};

} // namespace ckw

#endif // CKW_VALIDATION_TESTS_CLKERNELWRITERDECLARECONSTANTTILETEST_H
