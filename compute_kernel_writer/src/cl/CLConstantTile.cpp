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
#include "ckw/Error.h"
#include "ckw/TileInfo.h"

#include "src/Helpers.h"
#include "src/cl/CLHelpers.h"
#include "src/cl/CLConstantTile.h"

namespace ckw
{
CLConstantTile::CLConstantTile(const TileContainer &vals, DataType dt)
{
    const int32_t w = vals[0].size();
    const int32_t h = vals.size();

    _info.width(w);
    _info.height(h);
    _info.data_type(dt);

    validate_tile_info(_info);

    _vals = TileContainer(h, std::vector<std::string>(w));

    for(int32_t y = 0; y < h; ++y)
    {
        for(int32_t x = 0; x < w; ++x)
        {
            _vals[y][x] = vals[y][x];
        }
    }
}

TileVariable CLConstantTile::scalar(int32_t row, int32_t col) const
{
    // Clamp to nearest valid edge
    col = clamp(col, static_cast<int32_t>(0), _info.width() - 1);
    row = clamp(row, static_cast<int32_t>(0), _info.height() - 1);

    // We can use the vector method to retrieve the scalar variable stored in the constant tile
    return vector(row, col, 1);
}

TileVariable CLConstantTile::vector(int32_t row) const
{
    // Clamp to nearest valid edge
    row = clamp(row, static_cast<int32_t>(0), _info.height() - 1);

    return vector(row, 0, _info.width());
}

TileVariable CLConstantTile::vector(int32_t row, int32_t col_start, int32_t width) const
{
    // Validate the new vector length
    cl_validate_vector_length(width);

    // Clamp to nearest valid edge
    row = clamp(row, static_cast<int32_t>(0), _info.height() - 1);

    TileVariable t;
    t.desc.dt  = _info.data_type();
    t.desc.len = width;

    // The vector has the following form: ((data_typeN)(val0, val1,..., ValN-1))
    t.str = "((" + cl_get_variable_datatype_as_string(t.desc.dt, width) + ")";
    t.str += "(";

    int32_t col = col_start;
    for(; col < width - 1; ++col)
    {
        t.str += _vals[row][col];
        t.str += ", ";
    }
    t.str += _vals[row][col];
    t.str += "))";

    return t;
}

std::vector<TileVariable> CLConstantTile::all() const
{
    std::vector<TileVariable> vars;

    for(int32_t y = 0; y < _info.height(); ++y)
    {
        for(int32_t x = 0; x < _info.width(); ++x)
        {
            // We can use the vector method to retrieve all the scalar variables stored in the constant tile
            TileVariable t = vector(y, x, 1);
            vars.push_back(t);
        }
    }
    return vars;
}

bool CLConstantTile::is_assignable() const
{
    return false;
}
} // namespace ckw