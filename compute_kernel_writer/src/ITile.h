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
#ifndef COMPUTE_KERNEL_WRITER_SRC_ITILE_H
#define COMPUTE_KERNEL_WRITER_SRC_ITILE_H

#include "ckw/TileInfo.h"

#include <string>
#include <vector>

namespace ckw
{
/** Compute Kernel Writer tile container. It contains the variables stored in the tile as a string */
using TileContainer = std::vector<std::vector<std::string>>;

/** Tile descriptor which reports the underlying datatype and vector length */
struct TileVariableDescriptor
{
    DataType dt { DataType::Unknown };  /** Data type  */
    int32_t  len { 1 };                 /** Number of elements in a single variable. For example, 1 for scalar  */
};

/** Tile variable */
struct TileVariable
{
    std::string            str {""}; /** Tile variable as a string */
    TileVariableDescriptor desc {};  /** Tile value descriptor which reports the datatype and vector length */
};

/** Tile base class.
 *  A Tile is a collection of variables (either program variables or constants) used to express a 2D data.
 */
class ITile
{
public:
    virtual ~ITile() = default;
    /** Method to get all TileVariable objects
     *
     * @return a vector containing all @ref TileVariable objects
     */
    virtual std::vector<TileVariable> all() const = 0;
    /** Method to get the name of the tile.
     *
     * @return the name of the tile
     */
    std::string name() const
    {
        return _basename;
    }
    /** Method to get the tile info
     *
     * @return the @ref TileInfo
     */
    TileInfo info() const
    {
        return _info;
    }
    /** Method to know whether the tile is assignable or not.
     *  For example, a constant tile is not assignable.
     *
     * @return true if the tile is assignable
     */
    virtual bool is_assignable() const = 0;

protected:
    TileInfo    _info { DataType::Unknown };    // Tile info
    std::string _basename { "" };               // Tile name
};

/** Tile base class to store scalar variables.
 */
class IScalarTile : public ITile
{
public:
    virtual ~IScalarTile() = default;
    /** Method to get the scalar variable from a tile as a string
     * @param[in] row Tile row. If out-of-bound, the row is clamped to the nearest valid edge
     * @param[in] col Tile column. If out-of-bound, the column is clamped to the nearest valid edge
     *
     * @return the @ref TileVariable
     */
    virtual TileVariable scalar(int32_t row, int32_t col) const = 0;
};

/** Tile base class to store vector variables. It derives from IScalarTile since we can still access the scalar variable
 */
class IVectorTile : public IScalarTile
{
public:
    virtual ~IVectorTile() = default;
    /** Method to get the vector variable from a tile.
     *  The user can query the list of supported vector lengths through the supported_vector_lengths() method.
     *
     * @param[in] row Tile row. If out-of-bound, the row is clamped to the nearest valid edge
     *
     * @return the vector variable as a @ref TileVariable
     */
    virtual TileVariable vector(int32_t row) const = 0;
    /** Method to get a sub-vector variable. The length of the sub-vector must be supported by the derived IVectorTile class
     *
     * @param[in] row       Tile row. If out-of-bound, the row is clamped to the nearest valid edge
     * @param[in] col_start Tile starting column to get the sub-vector. If out-of-bound, the derived IVectorTile class may throw an assert.
     * @param[in] width     The width of the sub-vector. The width must be supported by the derived IVectorTile class and the last element must be in-bound.
     *
     * @return the vector variable as a @ref TileVariable
     */
    virtual TileVariable vector(int32_t row, int32_t col_start, int32_t width) const = 0;
    /** Method to get the supported vector length.
     *
     * @return a vector containing the supported vector lengths
     */
    virtual std::vector<int32_t> supported_vector_lengths() const = 0;
};
} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_SRC_ITILE_H */
