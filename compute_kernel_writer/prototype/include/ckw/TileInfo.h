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

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_TILEINFO_H
#define CKW_PROTOTYPE_INCLUDE_CKW_TILEINFO_H

#include "ckw/types/DataType.h"

#include <array>
#include <cstdint>

namespace ckw
{
// Constants to access the tile width and height in the TileShape
constexpr int32_t kTileWidthIdx  = 0;
constexpr int32_t kTileHeightIdx = 1;

/** Compute Kernel Writer tile shape. It is used to define the shape of the tile */
using TileShape = std::array<int32_t, 2>;

/** Compute Kernel Writer tile info */
class TileInfo
{
public:
    /** Constructor used to initialize a scalar variable with a given data type
     *
     * @param[in] dt Tile data type
     */
    TileInfo(DataType dt);

    /** Constructor used to initialize a vector with a given data type and vector length.
     *
     * @param[in] dt Tile data type
     * @param[in] w  Tile width (or vector length)
     */
    TileInfo(DataType dt, int32_t w);

    /** Constructor used to initialize a tile with a given data type and tile sizes.
     *
     * @param[in] dt Tile data type
     * @param[in] h  Tile height
     * @param[in] w  Tile width
     */
    TileInfo(DataType dt, int32_t h, int32_t w);

    /** Set width */
    TileInfo &width(int32_t w);

    /** Get width */
    int32_t width() const;

    /** Set height */
    TileInfo &height(int32_t h);

    /** Get height */
    int32_t height() const;

    /** Set data type */
    TileInfo &data_type(DataType dt);

    /** Get data type */
    DataType data_type() const;

private:
    DataType  _dt{DataType::Unknown};
    TileShape _shape{};
};

} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_INCLUDE_CKW_TILEINFO_H */
