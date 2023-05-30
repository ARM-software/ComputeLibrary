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

#include "ckw/Error.h"
#include "ckw/OperandBase.h"
#include "ckw/ScalarValue.h"
#include "ckw/TileInfo.h"

#include <vector>

namespace ckw
{

class Kernel;

/** Tile operand which can be either scalar, vector or 2D tile. */
class TileOperand : public OperandBase
{
public:
    /** Initialize a new instance of @ref TileOperand class with the tile information.
     *
     * @param[in] name      The name of the tile.
     * @param[in] tile_info The tile info.
     */
    TileOperand(const ::std::string &name, const TileInfo &tile_info);

    /** Initialize a new instance of @ref TileOperand for scalar variable.
     *
     * @param[in] name      The name of the tile.
     * @param[in] data_type The data type of the tile.
     */
    TileOperand(const ::std::string &name, DataType data_type);

    /** Initialize a new instance of @ref TileOperand for compile-time constant scalar variable.
     *
     * @param[in] name  The name of the tile.
     * @param[in] value The value of the tile.
     */
    TileOperand(const ::std::string &name, int32_t value);

    /** Initialize a new instance of @ref TileOperand for compile-time constant scalar variable.
     *
     * @param[in] name  The name of the tile.
     * @param[in] value The value of the tile.
     */
    TileOperand(const ::std::string &name, float value);

    /** Prohibit copy of tile operand. */
    TileOperand(const TileOperand &) = delete;

    /** Prohibit copy of tile operand. */
    TileOperand &operator=(const TileOperand &) = delete;

    /** (Internal use only) Create the implementation operand.
     *
     * @param[in] writer The implementation kernel writer.
     */
    virtual prototype::Operand create_impl_operand(prototype::IGpuKernelWriter *writer) const override;

    /** Get the tile info. */
    const TileInfo &tile_info() const;

    /** Get the data type of the tile. */
    virtual DataType data_type() const override;

    /** Get whether the tile is compile-time constant. */
    virtual bool is_constant() const override;

    /** Get whether the tile is a scalar value. */
    bool is_scalar() const;

    /** Get the scalar value of the tile.
     *
     * The tile must have the shape of 1, 1 (i.e. scalar).
     */
    ScalarValue scalar_value() const;

private:
    TileInfo    _info;
    ScalarValue _value{};
    bool        _constant;
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TILEOPERAND_H
