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

#ifndef CKW_VALIDATION_TESTS_CLKERNELWRITER_H
#define CKW_VALIDATION_TESTS_CLKERNELWRITER_H

#include "ckw/TileInfo.h"
#include "ckw/types/DataType.h"
#include "src/cl/CLKernelWriter.h"
#include "validation/tests/common/KernelWriterInterceptor.h"
#include "validation/tests/common/Common.h"

#include <vector>

namespace ckw
{

using CLKernelWriterDeclareTileConfig = std::tuple<DataType, int32_t, int32_t, std::string>;

class CLKernelWriterDeclareTileTest : public ITest
{
public:
    CLKernelWriterDeclareTileTest()
    {
        _configs = {
            {DataType::Fp32, 4, 4, "float4 G0__a_tile"},
            {DataType::Uint8, 4, 1, "uchar G0__a_tile"},
            {DataType::Int8, 4, 2, "char2 G0__a_tile"},
            {DataType::Bool, 9, 3, "bool3 G0__a_tile"},
            {DataType::Fp16, 4, 16, "half16 G0__a_tile"},
            {DataType::Uint32, 1, 8, "uint8 G0__a_tile"},
            {DataType::Uint16, 2, 3, "ushort3 G0__a_tile"},
        };
    }

    bool run() override
    {
        bool all_tests_passed = true;
        int32_t test_idx = 0;

        for(auto _config: _configs)
        {
            KernelWriterInterceptor<CLKernelWriter> writer;
            writer.start_capture_code();

            const DataType data_type = std::get<0>(_config);
            const int32_t height = std::get<1>(_config);
            const int32_t width = std::get<2>(_config);
            const std::string prefix = std::get<3>(_config);

            // expected output
            std::string expected_code = "";
            for(int32_t row = 0; row < height; ++row)
            {
                expected_code += prefix + ((height > 1) ? std::string("__") + std::to_string(row) : "") + ";\n";
            }

            TileInfo tile_info(data_type, height, width);
            writer.declare_tile("a_tile", tile_info);

            VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_idx++);
        }

        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLKernelWriterDeclareTileTest";
    }

private:
    std::vector<CLKernelWriterDeclareTileConfig> _configs {};
};

} // namespace ckw

#endif /* CKW_VALIDATION_TESTS_CLKERNELWRITER_H */
