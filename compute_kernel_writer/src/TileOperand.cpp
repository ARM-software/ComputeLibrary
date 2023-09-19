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

#include "ckw/TileOperand.h"
#include "ckw/Error.h"
#include "src/ITile.h"

namespace ckw
{

TileOperand::TileOperand(ITile &tile)
    : _tile(&tile), _row_start(0), _row_end(tile.info().height()), _col_start(0), _col_end(tile.info().width())
{
}

TileOperand::TileOperand(const TileOperand &operand, int32_t row_start, int32_t row_end, int32_t col_start, int32_t col_end)
    : _tile(operand._tile), _row_start(row_start), _row_end(row_end), _col_start(col_start), _col_end(col_end)
{
    CKW_ASSERT(row_start >= 0 && row_start < _tile->info().height());
    CKW_ASSERT(row_end > row_start && row_end <= _tile->info().height());
    CKW_ASSERT(col_start >= 0 && col_start < _tile->info().width());
    CKW_ASSERT(col_end > col_start && col_end <= _tile->info().width());
}

TileOperand TileOperand::tile(int32_t row_start, int32_t row_end, int32_t col_start, int32_t col_end) const
{
    CKW_ASSERT(row_start >= 0 && _row_start + row_start < _row_end);
    CKW_ASSERT(row_end > row_start && _row_start + row_end <= _row_end);
    CKW_ASSERT(col_start >= 0 && _col_start + col_start < _col_end);
    CKW_ASSERT(col_end > col_start && _col_start + col_end <= _col_end);

    return TileOperand(*this, _row_start + row_start, _row_start + row_end, _col_start + col_start, _col_start + col_end);
}

TileOperand TileOperand::row(int32_t row) const
{
    CKW_ASSERT(row >= 0 && _row_start + row < _row_end);

    return tile(_row_start + row, _row_start + row + 1, _col_start, _col_end);
}

TileOperand TileOperand::scalar(int32_t row, int32_t col) const
{
    CKW_ASSERT(row >= 0 && _row_start + row < _row_end);
    CKW_ASSERT(col >= 0 && _col_start + col < _col_end);

    return tile(_row_start + row, _row_start + row + 1, _col_start + col, _col_start + col + 1);
}

} // namespace ckw
