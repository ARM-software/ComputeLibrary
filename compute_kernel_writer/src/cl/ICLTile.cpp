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

#include "src/cl/CLHelpers.h"
#include "src/cl/ICLTile.h"

#include <vector>

namespace ckw
{
std::vector<int32_t> ICLTile::supported_vector_lengths() const
{
    return std::vector<int32_t> {1, 2, 3, 4, 8, 16};
}

void ICLTile::validate_tile_info(const TileInfo &info) const
{
    if(cl_validate_vector_length(info.width()))
    {
        COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("Unsupported TileInfo width");
    }

    if(info.data_type() == DataType::Unknown)
    {
        COMPUTE_KERNEL_WRITER_ERROR_ON_MSG("DataType::Unknown is not supported");
    }
}
} // namespace ckw