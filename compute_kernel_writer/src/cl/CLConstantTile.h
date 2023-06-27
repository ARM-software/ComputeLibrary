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
#ifndef COMPUTE_KERNEL_WRITER_SRC_CL_CLCONSTANTTILE_H
#define COMPUTE_KERNEL_WRITER_SRC_CL_CLCONSTANTTILE_H

#include "src/ITile.h"
#include "src/cl/ICLTile.h"

namespace ckw
{
// Forward declarations
class TileInfo;

/** OpenCL specific constant tile */
class CLConstantTile : public ICLTile
{
public:
    /** Constructor
     *
     * @note A constant tile does not need a name since this object does not return variable's name but rather
     *       values stored as string type
     *
     * @param[in] vals The tile container with the constant values as std::string
     * @param[in] dt   Datatype of the values stored in the tile container
    */
    CLConstantTile(const TileContainer &vals, DataType dt);

    // Inherited method overridden
    TileVariable              scalar(int32_t row, int32_t col) const override;

    TileVariable              vector(int32_t row) const override;

    TileVariable              vector(int32_t row, int32_t col_start, int32_t width) const override;

    std::vector<TileVariable> all() const override;

    bool                      is_assignable() const override;

private:
    TileContainer _vals{};
};
} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_SRC_CL_CLCONSTANTTILE_H */
