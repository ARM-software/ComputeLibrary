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

#ifndef CKW_INCLUDE_CKW_TILEOPERAND_H
#define CKW_INCLUDE_CKW_TILEOPERAND_H

#include <cstdint>

namespace ckw
{

class KernelWriter;
class TensorOperand;
class ITile;
class TileInfo;

/** A tile operand refers to a tile object that can be used for kernel writing. */
class TileOperand
{
public:
    // The constructor and _tile field is completely hidden from the public API to avoid any misuse.
    // Only kernel writer and tensor operand classes create and interact with tile operand hence we allow them to access this field.
    friend class KernelWriter;
    friend class TensorOperand;

    /** Create an empty tile operand.
     *
     * The new tile operand doesn't refer to any tile therefore it is not useable.
     */
    TileOperand();

    /** Check if the tile operand contains a tile and therefore useable. */
    bool is_valid() const;

    /** Get the tile info. */
    const TileInfo &tile_info() const;

    /** Get a row vector of the current tile operand.
     *
     * @param[in] row The index of the row to be accessed in the current tile operand.
     *
     * @return A new tile operand referring to a row of the current tile operand.
     */
    TileOperand row(int32_t row) const;

    /** Get a scalar element of the current tile operand.
     *
     * @param[in] row The index of the row to be accessed in the current tile operand.
     * @param[in] col The index of the column to be accessed in the current tile operand.
     *
     * @return A new tile operand referring to a scalar element of the current tile operand.
     */
    TileOperand scalar(int32_t row, int32_t col) const;

private:
    // These are hidden from the public API to avoid any misuse.

    /** Initialize a new instance of @ref TileOperand class for the given tile. */
    TileOperand(ITile &tile);

    /** Initialize a new instance of @ref TileOperand class that is the sub-tile of the given tile. */
    TileOperand(const TileOperand &operand, int32_t row_start, int32_t row_end, int32_t col_start, int32_t col_end);

    /** Get a sub-tile of the current tile operand.
     *
     * The range of rows and columns is defined by pairs of start and end indices, inclusive lower and exclusive upper.
     * In other words, any row and column indices satisfying the following conditions will be part of the sub-tile:
     *
     *   row_start <= row_index < row_end
     *   col_start <= col_index < col_end
     *
     * @param[in] row_start The start index of the row range.
     * @param[in] row_end   The end index of the row range.
     * @param[in] col_start The start index of the column range.
     * @param[in] col_end   The end index of the column range.
     *
     * @return A new tile operand refering to the same tile but with the new active area.
     */
    TileOperand tile(int32_t row_start, int32_t row_end, int32_t col_start, int32_t col_end) const;

    ITile *_tile;

    int32_t _row_start;
    int32_t _row_end;
    int32_t _col_start;
    int32_t _col_end;
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TILEOPERAND_H
