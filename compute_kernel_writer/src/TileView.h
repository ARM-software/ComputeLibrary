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

#ifndef CKW_SRC_TILEVIEW_H
#define CKW_SRC_TILEVIEW_H

#include "ckw/Error.h"
#include "ckw/types/DataType.h"
#include "src/ITile.h"

#include <cstdint>

namespace ckw
{

/** A rectangular active area of a tile. */
class TileArea
{
public:
    /** Create a new tile rectangular active area.
     *
     * The range of rows and columns is defined by pairs of start and end indices, inclusive lower and exclusive upper.
     * In other word, any row and column indices satisfied the following conditions will be part of the active area:
     *
     *   row_start <= row_index < row_end
     *   col_start <= col_index < col_end
     *
     * @param[in] row_start The start index of the row range.
     * @param[in] row_end   The end index of the row range.
     * @param[in] col_start The start index of the column range.
     * @param[in] col_end   The end index of the column range.
     */
    TileArea(int32_t row_start, int32_t row_end, int32_t col_start, int32_t col_end);

    /** Get the start row index. */
    int32_t row_start() const;

    /** Get the end row (exclusive) index. */
    int32_t row_end() const;

    /** Get the start column index. */
    int32_t col_start() const;

    /** Get the end column (exclusive) index. */
    int32_t col_end() const;

private:
    int32_t _row_start;
    int32_t _row_end;
    int32_t _col_start;
    int32_t _col_end;
};

/** A rectangular view of a tile. */
template <typename T>
class TileView
{
public:
    /** Create a tile view that refers to the whole tile.
     *
     * @param[in] tile The tile object.
     */
    TileView(const T &tile)
        : _tile(&tile), _area(0, tile.info().height(), 0, tile.info().width())
    {
    }

    /** Create a new rectangular view of the given tile.
     *
     * @param[in] tile The tile object.
     * @param[in] area The rectangular active area.
     */
    TileView(const T &tile, const TileArea &area)
        : _tile(&tile), _area(area)
    {
    }

    /** Get the tile object.
     *
     * The caller must guarantee that the tile view refers to the whole tile.
     */
    const T &full_tile() const
    {
        CKW_ASSERT(is_full_tile());

        return *_tile;
    }

    /** Get the data type of the tile. */
    DataType data_type() const
    {
        return _tile->info().data_type();
    }

    /** Get the start row index. */
    int32_t row_start() const
    {
        return _area.row_start();
    }

    /** Get the end row index. */
    int32_t row_end() const
    {
        return _area.row_end();
    }

    /** Get the start column index. */
    int32_t col_start() const
    {
        return _area.col_start();
    }

    /** Get the end column index. */
    int32_t col_end() const
    {
        return _area.col_end();
    }

    /** Get the height of the tile view. */
    int32_t height() const
    {
        return _area.row_end() - _area.row_start();
    }

    /** Get the width of the tile view. */
    int32_t width() const
    {
        return _area.col_end() - _area.col_start();
    }

    /** See @ref IVectorAccess::vector. */
    TileVariable vector(int32_t row) const
    {
        return _tile->vector(row_start() + row, col_start(), width());
    }

    /** See @ref IScalarAccess::scalar. */
    TileVariable scalar(int32_t row, int32_t col) const
    {
        return _tile->scalar(row_start() + row, col_start() + col);
    }

    /** Get the name of the tile. */
    const std::string &name() const
    {
        return _tile->name();
    }

    /** Get whether the tile view is a scalar element. */
    bool is_scalar() const
    {
        return height() == 1 && width() == 1;
    }

    /** Get whether the tile view refers to the whole tile. */
    bool is_full_tile() const
    {
        return row_start() == 0 && row_end() == _tile->info().height() && col_start() == 0 && col_end() == _tile->info().width();
    }

private:
    const T *_tile;
    TileArea _area;
};

} // namespace ckw

#endif // CKW_SRC_TILEVIEW_H
