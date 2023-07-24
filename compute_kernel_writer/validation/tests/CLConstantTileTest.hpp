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

#ifndef CKW_TESTS_CLCONSTANTTILETEST_HPP
#define CKW_TESTS_CLCONSTANTTILETEST_HPP

#include "common/Common.h"
#include "src/Helpers.h"
#include "src/cl/CLHelpers.h"
#include "src/cl/CLTile.h"

#include <random>
#include <string>
#include <vector>

namespace ckw
{
class CLConstantTileInternalValuesTest : public ITest
{
public:
    CLConstantTileInternalValuesTest()
    {
        _values.push_back({ { "1.2", "3.5" },
                            { "4.2", "1.3" } });
        _values.push_back({ { "1.2" } });
        _values.push_back({ { "1.2", "6.9" } });
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        int32_t test_idx = 0;
        for(const auto &test : _values)
        {
            const CLTile  tile(test, DataType::Fp16);
            const auto    vars     = tile.all();
            const int32_t num_vars = vars.size();
            const int32_t width    = tile.info().width();

            for(int32_t y = 0; y < num_vars; ++y)
            {
                const int32_t     col               = y % width;
                const int32_t     row               = y / width;
                const std::string expected_var_name = "((half)(" + test[row][col] + "))";
                const std::string actual_var_name   = vars[y].str;
                VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
            }
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLConstantTileInternalValuesTest";
    }

private:
    std::vector<TileContainer> _values{};
};

class CLConstantTileAccessScalarVariableBroadcastXTest : public ITest
{
public:
    const std::string tile_name = "src";
    const int32_t     height    = 8;
    const DataType    dt        = DataType::Fp16;

    CLConstantTileAccessScalarVariableBroadcastXTest()
    {
        _width.push_back(1);
        _width.push_back(2);
        _width.push_back(3);

        _x_coord.push_back(4);
        _x_coord.push_back(5);
        _x_coord.push_back(6);

        _y_coord.push_back(1);
        _y_coord.push_back(3);
        _y_coord.push_back(2);
    }

    bool run() override
    {
        VALIDATE_ON_MSG(_width.size() == _y_coord.size(), "The number of widths and y-coords does not match");
        VALIDATE_ON_MSG(_x_coord.size() == _y_coord.size(), "The number of x-coords and y-coords does not match");

        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const size_t num_coords = _x_coord.size();

        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dist(-1, 1);

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_coords; ++i)
        {
            const int32_t width   = _width[i];
            const int32_t x_coord = _x_coord[i];
            const int32_t y_coord = _y_coord[i];

            const int32_t x_coord_clamped = clamp(x_coord, static_cast<int32_t>(0), width - 1);

            TileContainer container = TileContainer(height, std::vector<std::string>(width));

            for(int32_t row = 0; row < height; ++row)
            {
                for(int32_t col = 0; col < width; ++col)
                {
                    container[row][col] = std::to_string(dist(gen));
                }
            }

            const CLTile tile(container, dt);

            const TileVariable var = tile.scalar(y_coord, x_coord);

            const std::string actual_var_name   = var.str;
            const std::string expected_var_name = "((half)(" + container[y_coord][x_coord_clamped] + "))";

            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLConstantTileAccessScalarVariableBroadcastXTest";
    }

private:
    std::vector<int32_t> _width{};
    std::vector<int32_t> _x_coord{};
    std::vector<int32_t> _y_coord{};
};

class CLConstantTileAccessScalarVariableBroadcastYTest : public ITest
{
public:
    const std::string tile_name = "src";
    const int32_t     width     = 8;
    const DataType    dt        = DataType::Fp16;

    CLConstantTileAccessScalarVariableBroadcastYTest()
    {
        _height.push_back(1);
        _height.push_back(2);
        _height.push_back(3);

        _x_coord.push_back(4);
        _x_coord.push_back(5);
        _x_coord.push_back(6);

        _y_coord.push_back(3);
        _y_coord.push_back(4);
        _y_coord.push_back(5);
    }

    bool run() override
    {
        VALIDATE_ON_MSG(_height.size() == _y_coord.size(), "The number of widths and y-coords does not match");
        VALIDATE_ON_MSG(_x_coord.size() == _y_coord.size(), "The number of x-coords and y-coords does not match");

        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dist(-1, 1);

        const size_t num_coords = _x_coord.size();

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_coords; ++i)
        {
            const int32_t height  = _height[i];
            const int32_t x_coord = _x_coord[i];
            const int32_t y_coord = _y_coord[i];

            const int32_t y_coord_clamped = clamp(y_coord, static_cast<int32_t>(0), height - 1);

            TileContainer container = TileContainer(height, std::vector<std::string>(width));

            for(int32_t row = 0; row < height; ++row)
            {
                for(int32_t col = 0; col < width; ++col)
                {
                    container[row][col] = std::to_string(dist(gen));
                }
            }

            const CLTile tile(container, dt);

            const TileVariable var = tile.scalar(y_coord, x_coord);

            const std::string actual_var_name   = var.str;
            const std::string expected_var_name = "((half)(" + container[y_coord_clamped][x_coord] + "))";

            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLConstantTileAccessScalarVariableBroadcastYTest";
    }

private:
    std::vector<int32_t> _height{};
    std::vector<int32_t> _x_coord{};
    std::vector<int32_t> _y_coord{};
};

class CLConstantTileAccessVectorVariablesTest : public ITest
{
public:
    const DataType dt = DataType::Fp16;

    CLConstantTileAccessVectorVariablesTest()
    {
        _values.push_back({ { "1.2", "3.5" },
                            { "4.2", "1.3" } });
        _values.push_back({ { "1.2" } });
        // Mix variable names and values
        _values.push_back({ { "1.2", "acc", "8.7", "9.3", "ratio", "2.9", "1.7", "0.3" } });
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        int32_t test_idx = 0;

        for(const auto &test : _values)
        {
            const CLTile  tile(test, dt);
            const int32_t width  = tile.info().width();
            const int32_t height = tile.info().height();

            for(int32_t row = 0; row < height; ++row)
            {
                std::string expected_var_name = "((";
                expected_var_name += cl_get_variable_datatype_as_string(dt, width);
                expected_var_name += ")(";

                int32_t col = 0;
                for(; col < width - 1; ++col)
                {
                    expected_var_name += test[row][col];
                    expected_var_name += ", ";
                }

                expected_var_name += test[row][col];
                expected_var_name += "))";

                const std::string actual_var_name = tile.vector(row).str;
                VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
            }
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLConstantTileAccessVectorVariablesTest";
    }

private:
    std::vector<TileContainer> _values{};
};

class CLConstantTileAccessSubVectorVariablesTest : public ITest
{
public:
    const DataType dt = DataType::Fp16;

    CLConstantTileAccessSubVectorVariablesTest()
    {
        _values.push_back({ { "1.2", "acc", "8.7", "9.3", "ratio", "2.9", "1.7", "0.3" } });
        _subwidths.push_back(1);
        _subwidths.push_back(2);
        _subwidths.push_back(3);
        _subwidths.push_back(4);
        _offsets.push_back(1);
        _offsets.push_back(3);
        _offsets.push_back(4);
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        size_t test_idx = 0;

        for(auto &test : _values)
        {
            for(auto &col_start : _offsets)
            {
                for(auto &subwidth : _subwidths)
                {
                    const CLTile  tile(test, dt);
                    const int32_t height = tile.info().height();

                    for(int32_t row = 0; row < height; ++row)
                    {
                        std::string expected_var_name = "((";
                        expected_var_name += cl_get_variable_datatype_as_string(dt, subwidth);
                        expected_var_name += ")(";

                        int32_t col = col_start;
                        for(; col < subwidth - 1; ++col)
                        {
                            expected_var_name += test[row][col];
                            expected_var_name += ", ";
                        }

                        expected_var_name += test[row][col];
                        expected_var_name += "))";

                        const std::string actual_var_name = tile.vector(row, col_start, subwidth).str;
                        VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed,
                                      test_idx++);
                    }
                }
            }
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLConstantTileAccessSubVectorVariablesTest";
    }

private:
    std::vector<TileContainer> _values{};
    std::vector<int32_t>       _subwidths{};
    std::vector<int32_t>       _offsets{};
};

} // namespace ckw

#endif // CKW_TESTS_CLCONSTANTTILETEST_HPP
