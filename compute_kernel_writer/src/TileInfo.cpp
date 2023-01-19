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

#include "ckw/TileInfo.h"

namespace ckw
{
TileInfo::TileInfo(DataType dt)
    : _dt(dt), _shape({{1, 1}})
{
}

TileInfo::TileInfo(DataType dt, int32_t w)
    : _dt(dt), _shape({{w, 1}})
{
}

TileInfo::TileInfo(DataType dt, int32_t w, int32_t h)
    : _dt(dt), _shape({{w, h}})
{
}

TileInfo &TileInfo::width(int32_t w)
{
    _shape[kTileWidthIdx] = w;
    return *this;
}

int32_t TileInfo::width() const
{
    return _shape[kTileWidthIdx];
}

TileInfo &TileInfo::height(int32_t h)
{
    _shape[kTileHeightIdx] = h;
    return *this;
}

int32_t TileInfo::height() const
{
    return _shape[kTileHeightIdx];
}

TileInfo &TileInfo::data_type(DataType dt)
{
    _dt = dt;
    return *this;
}

DataType TileInfo::data_type() const
{
    return _dt;
}
} // namespace ckw
