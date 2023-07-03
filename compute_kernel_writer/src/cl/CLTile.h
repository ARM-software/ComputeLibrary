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
#ifndef COMPUTE_KERNEL_WRITER_SRC_CL_CLTILE_H
#define COMPUTE_KERNEL_WRITER_SRC_CL_CLTILE_H

#include "src/ITile.h"
#include "src/cl/ICLTile.h"
#include "ckw/ITileOperand.h"

#include <string>

namespace ckw
{
// Forward declarations
class TileInfo;

/** OpenCL specific tile */
class CLTile : public ICLTile, public ITileOperand
{
public:
    /** Constructor
     *
     * @param[in] name Tile name
     * @param[in] info Tile info
    */
    CLTile(const std::string &name, const TileInfo &info);

    // Inherited method overridden
    TileVariable scalar(int32_t row, int32_t col) const override;

    TileVariable vector(int32_t row) const override;

    TileVariable vector(int32_t row, int32_t col_start, int32_t width) const override;

    std::vector<TileVariable> all() const override;

    bool is_assignable() const override;

private:
    std::string create_var_name(int32_t row) const;
};
} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_SRC_CL_CLTILE_H */
