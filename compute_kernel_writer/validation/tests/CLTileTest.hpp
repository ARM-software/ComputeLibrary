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

#ifndef COMPUTE_KERNEL_WRITER_TESTS_CLTILETEST_HPP
#define COMPUTE_KERNEL_WRITER_TESTS_CLTILETEST_HPP

#include "common/Common.h"
#include "src/Helpers.h"
#include "src/cl/CLTile.h"

#include <string>
#include <vector>

namespace ckw
{
class CLTileInternalVariableNamesTest : public ITest
{
public:
    const int32_t  width  = 4;
    const int32_t  height = 4;
    const DataType dt     = DataType::Fp32;

    CLTileInternalVariableNamesTest()
    {
        _tile_name.push_back("dst");
        _tile_name.push_back("_G0_dst");
        _tile_name.push_back("_SRC");
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const TileInfo info(dt, height, width);

        int32_t test_idx = 0;
        for(const auto &tile_name : _tile_name)
        {
            const CLTile tile(tile_name, info);
            const auto   vars = tile.all();

            for(int32_t y = 0; y < height; ++y)
            {
                const std::string expected_var_name = tile_name + "_" + std::to_string(y);
                const std::string actual_var_name   = vars[y].str;
                VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
            }
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTileInternalVariableNamesTest";
    }

private:
    std::vector<std::string> _tile_name{};
};

class CLTileInternalNumVariablesTest : public ITest
{
public:
    CLTileInternalNumVariablesTest()
    {
        _width.push_back(4);
        _width.push_back(1);
        _width.push_back(16);

        _height.push_back(1);
        _height.push_back(5);
        _height.push_back(3);
    }

    bool run() override
    {
        VALIDATE_ON_MSG(_width.size() == _height.size(), "The number of widths and heights does not match");

        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const size_t num_dims = _width.size();

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_dims; ++i)
        {
            const int32_t  width  = _width[i];
            const int32_t  height = _height[i];
            const TileInfo info(DataType::Fp32, height, width);
            const CLTile   tile("src", info);
            const auto     vars     = tile.all();
            const int32_t  num_vars = vars.size();

            // We expect the number of variables to match the heigth of the tile
            VALIDATE_TEST(num_vars == height, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTileInternalNumVariablesTest";
    }

private:
    std::vector<int32_t> _width{};
    std::vector<int32_t> _height{};
};

class CLTileAccessScalarVariableTest : public ITest
{
public:
    const std::string tile_name = "src";
    const int32_t     width     = 16;
    const int32_t     height    = 8;
    const DataType    dt        = DataType::Fp32;

    CLTileAccessScalarVariableTest()
    {
        _x_coord.push_back(4);
        _x_coord.push_back(1);
        _x_coord.push_back(15);
        _x_coord.push_back(10);

        _y_coord.push_back(1);
        _y_coord.push_back(5);
        _y_coord.push_back(3);
        _y_coord.push_back(4);
    }

    bool run() override
    {
        const TileInfo info(dt, height, width);
        const CLTile   tile(tile_name, info);

        VALIDATE_ON_MSG(_x_coord.size() == _y_coord.size(), "The number of x-coords and y-coords does not match");

        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        const size_t num_coords = _x_coord.size();

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_coords; ++i)
        {
            const int32_t x_coord = _x_coord[i];
            const int32_t y_coord = _y_coord[i];

            const TileVariable var = tile.scalar(y_coord, x_coord);

            const std::string actual_var_name   = var.str;
            std::string       expected_var_name = tile_name;
            expected_var_name += "_" + std::to_string(y_coord);
            expected_var_name += ".s" + dec_to_hex_as_string(x_coord);

            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTileAccessScalarVariableTest";
    }

private:
    std::vector<int32_t> _x_coord{};
    std::vector<int32_t> _y_coord{};
};

class CLTileAccessScalarVariableBroadcastXTest : public ITest
{
public:
    const std::string tile_name = "src";
    const int32_t     height    = 8;
    const DataType    dt        = DataType::Fp32;

    CLTileAccessScalarVariableBroadcastXTest()
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

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_coords; ++i)
        {
            const int32_t width   = _width[i];
            const int32_t x_coord = _x_coord[i];
            const int32_t y_coord = _y_coord[i];

            const int32_t x_coord_clamped = clamp(x_coord, static_cast<int32_t>(0), width - 1);

            const TileInfo info(dt, height, width);
            const CLTile   tile(tile_name, info);

            const TileVariable var = tile.scalar(y_coord, x_coord);

            const std::string actual_var_name   = var.str;
            std::string       expected_var_name = tile_name;
            expected_var_name += "_" + std::to_string(y_coord);
            if(width != 1)
            {
                expected_var_name += ".s" + dec_to_hex_as_string(x_coord_clamped);
            }

            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTileAccessScalarVariableBroadcastXTest";
    }

private:
    std::vector<int32_t> _width{};
    std::vector<int32_t> _x_coord{};
    std::vector<int32_t> _y_coord{};
};

class CLTileAccessScalarVariableBroadcastYTest : public ITest
{
public:
    const std::string tile_name = "src";
    const int32_t     width     = 8;
    const DataType    dt        = DataType::Fp32;

    CLTileAccessScalarVariableBroadcastYTest()
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

        const size_t num_coords = _x_coord.size();

        int32_t test_idx = 0;
        for(size_t i = 0; i < num_coords; ++i)
        {
            const int32_t height  = _height[i];
            const int32_t x_coord = _x_coord[i];
            const int32_t y_coord = _y_coord[i];

            const int32_t y_coord_clamped = clamp(y_coord, static_cast<int32_t>(0), height - 1);

            const TileInfo info(dt, height, width);
            const CLTile   tile(tile_name, info);

            const TileVariable var = tile.scalar(y_coord, x_coord);

            const std::string actual_var_name   = var.str;
            std::string       expected_var_name = tile_name;
            if(height != 1)
            {
                expected_var_name += "_" + std::to_string(y_coord_clamped);
            }

            if(width != 1)
            {
                expected_var_name += ".s" + dec_to_hex_as_string(x_coord);
            }

            VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTileAccessScalarVariableBroadcastYTest";
    }

private:
    std::vector<int32_t> _height{};
    std::vector<int32_t> _x_coord{};
    std::vector<int32_t> _y_coord{};
};

class CLTileAccessVectorVariablesTest : public ITest
{
public:
    const std::string tile_name = "src";
    const int32_t     width     = 8;
    const DataType    dt        = DataType::Fp32;

    CLTileAccessVectorVariablesTest()
    {
        _heights.push_back(1);
        _heights.push_back(2);
        _heights.push_back(3);
    }

    bool run() override
    {
        // The status of this variable can change in VALIDATE_TEST()
        bool all_tests_passed = true;

        int32_t test_idx = 0;
        for(const auto &height : _heights)
        {
            const TileInfo info(dt, height, width);
            const CLTile   tile(tile_name, info);

            for(int32_t row = 0; row < height; ++row)
            {
                const TileVariable var = tile.vector(row);

                const std::string actual_var_name   = var.str;
                std::string       expected_var_name = tile_name;
                if(height != 1)
                {
                    expected_var_name += "_" + std::to_string(row);
                }

                VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
            }
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTileAccessVectorVariablesTest";
    }

private:
    std::vector<int32_t> _heights{};
};

class CLTileAccessSubVectorVariablesTest : public ITest
{
public:
    const std::string tile_name = "src";
    const int32_t     width     = 8;
    const int32_t     height    = 3;
    const DataType    dt        = DataType::Fp32;

    CLTileAccessSubVectorVariablesTest()
    {
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

        for(auto &col_start : _offsets)
        {
            for(const auto &subwidth : _subwidths)
            {
                const TileInfo info(dt, height, width);
                const CLTile   tile(tile_name, info);

                for(int32_t row = 0; row < height; ++row)
                {
                    std::string expected_var_name = tile_name;
                    if(height != 1)
                    {
                        expected_var_name += "_" + std::to_string(row);
                    }

                    if(width != 1)
                    {
                        expected_var_name += ".s";
                    }

                    int32_t col = col_start;
                    for(; col < col_start + subwidth - 1; ++col)
                    {
                        if(width != 1)
                        {
                            expected_var_name += dec_to_hex_as_string(col);
                        }
                    }

                    if(width != 1)
                    {
                        expected_var_name += dec_to_hex_as_string(col);
                    }

                    const std::string actual_var_name = tile.vector(row, col_start, subwidth).str;
                    VALIDATE_TEST(actual_var_name.compare(expected_var_name) == 0, all_tests_passed, test_idx++);
                }
            }
        }
        return all_tests_passed;
    }

    std::string name() override
    {
        return "CLTileAccessSubVectorVariablesTest";
    }

private:
    std::vector<int32_t> _subwidths{};
    std::vector<int32_t> _offsets{};
};
} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_TESTS_CLTILETEST_HPP */
