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

#ifndef CKW_VALIDATION_SRC_TESTS_CLKERNELWRITERSUBTILETEST_H
#define CKW_VALIDATION_SRC_TESTS_CLKERNELWRITERSUBTILETEST_H

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

class CLKernelWriterSubTileTest : public ITest
{
public:
    CLKernelWriterSubTileTest()
    {
        // These are the definitions of the tiles involving in the writing actions.
        //
        // Structure:
        //   * List of tiles:
        //     - Tile full height.
        //     - Tile full width.
        //     - Tile view access type (full tile, vector, scalar).
        //     - Tile view start row.
        //     - Tile view start column.
        //     - The tile name.

        // Vector access.
        _tests.push_back(
            { { { 1, 4, AccessType::Vector, 0, 0, "{tile_name}" }, //
                { 4, 4, AccessType::Vector, 2, 0, "{tile_name}__2" },
                { 1, 4, AccessType::Full, 0, 0, "{tile_name}" },
                { 4, 4, AccessType::Vector, 3, 0, "{tile_name}__3" } } });

        // Scalar access.
        _tests.push_back(
            { { { 1, 1, AccessType::Full, 0, 0, "{tile_name}" }, //
                { 4, 8, AccessType::Scalar, 2, 4, "{tile_name}__2.s4" },
                { 1, 16, AccessType::ScalarOfVector, 0, 10, "{tile_name}.sA" },
                { 1, 1, AccessType::Scalar, 0, 0, "{tile_name}" } } });

        // These are the definitions of the writing actions.
        //
        // Structure:
        //   * Writing function.
        //   * Whether this function only works with scalar value.
        //   * Expected code format.

        _actions.push_back(
            { [](CLKernelWriter &writer, const std::vector<TileOperand> &args)
              {
                  writer.op_assign(args.at(0), args.at(1));
              },
              false,
              "{op0} = {op1};\n" });

        _actions.push_back(
            { [](CLKernelWriter &writer, const std::vector<TileOperand> &args)
              {
                  writer.op_unary(args.at(0), UnaryOp::Sqrt, args.at(1));
              },
              false,
              "{op0} = sqrt({op1});\n" });

        _actions.push_back(
            { [](CLKernelWriter &writer, const std::vector<TileOperand> &args)
              {
                  writer.op_binary(args.at(0), BinaryOp::Add, args.at(1), args.at(2));
              },
              false,
              "{op0} = {op1} + {op2};\n" });

        _actions.push_back(
            { [](CLKernelWriter &writer, const std::vector<TileOperand> &args)
              {
                  writer.op_ternary(args.at(0), TernaryOp::Clamp, args.at(1), args.at(2), args.at(3));
              },
              false,
              "{op0} = clamp({op1}, {op2}, {op3});\n" });

        _actions.push_back(
            { [](CLKernelWriter &writer, const std::vector<TileOperand> &args)
              {
                  writer.op_if(args.at(0), BinaryOp::Greater, args.at(1), [] {});
              },
              true,
              "if ({op0} > {op1})\n{\n}\n" });
    }

    bool run() override
    {
        bool    all_tests_passed = true;
        int32_t test_idx         = 0;

        KernelWriterInterceptor<CLKernelWriter> writer;

        for(size_t test_no = 0; test_no < _tests.size(); ++test_no)
        {
            const TestInfo &test = _tests.at(test_no);

            // Declare all the tiles and get the full name of those tile operand.
            std::vector<TileOperand> tiles;
            std::vector<std::string> expected_tiles_name;

            for(size_t operand_no = 0; operand_no < test.operands.size(); ++operand_no)
            {
                const TestOperand &operand = test.operands.at(operand_no);
                std::string        name    = "test" + std::to_string(test_no) + "_op" + std::to_string(operand_no);

                const TileOperand full_tile = writer.declare_tile(name, TileInfo(DataType::Fp32, operand.height, operand.width));

                switch(operand.access_type)
                {
                    case AccessType::Full:
                        tiles.emplace_back(full_tile);
                        break;

                    case AccessType::Vector:
                        tiles.emplace_back(full_tile.row(operand.start_row));
                        break;

                    case AccessType::Scalar:
                        tiles.emplace_back(full_tile.scalar(operand.start_row, operand.start_col));
                        break;

                    case AccessType::ScalarOfVector:
                        tiles.emplace_back(full_tile.row(operand.start_row).scalar(0, operand.start_col));
                        break;

                    default:
                        CKW_THROW_MSG("Unsupported access type!");
                }

                expected_tiles_name.push_back("G0__" + name);
            }

            // Try each writing action using the newly declared tiles.
            for(const TestAction &action : _actions)
            {
                if(action.scalar_only &&                                     //
                   (test.operands.at(0).access_type != AccessType::Scalar && //
                    (test.operands.at(0).height != 1 || test.operands.at(0).width != 1)))
                {
                    continue;
                }

                writer.start_capture_code();

                action.write(writer, tiles);

                // The expected code is constructed from the format strings.
                std::string expected_code = action.expected_code;

                for(size_t operand_no = 0; operand_no < test.operands.size(); ++operand_no)
                {
                    const TestOperand &operand = test.operands.at(operand_no);

                    const std::string op_name = search_and_replace(operand.name, "{tile_name}", expected_tiles_name.at(operand_no));
                    expected_code             = search_and_replace(expected_code, "{op" + std::to_string(operand_no) + "}", op_name);
                }

                VALIDATE_TEST(writer.check_added_code(expected_code), all_tests_passed, test_idx++);
            }
        }

        return all_tests_passed;
    }

    std::string search_and_replace(const std::string &src, const std::string &search, const std::string &replace)
    {
        std::string result = src;

        size_t idx = 0;

        while(true)
        {
            idx = result.find(search, idx);

            if(idx == std::string::npos)
            {
                break;
            }

            result = result.replace(idx, search.size(), replace);
        }

        return result;
    }

    std::string name() override
    {
        return "CLKernelWriterSubTileTest";
    }

private:
    enum class AccessType
    {
        Full,
        Vector,
        Scalar,
        ScalarOfVector,
    };

    struct TestOperand
    {
        int32_t height;
        int32_t width;

        AccessType access_type;
        int32_t    start_row;
        int32_t    start_col;

        std::string name;
    };

    struct TestInfo
    {
        std::vector<TestOperand> operands;
    };

    struct TestAction
    {
        std::function<void(CLKernelWriter &, const std::vector<TileOperand> &)> write;

        bool        scalar_only;
        std::string expected_code;
    };

    std::vector<TestInfo>   _tests{};
    std::vector<TestAction> _actions{};
};

} // namespace ckw

#endif // CKW_VALIDATION_SRC_TESTS_CLKERNELWRITERSUBTILETEST_H
