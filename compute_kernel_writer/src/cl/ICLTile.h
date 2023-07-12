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
#ifndef COMPUTE_KERNEL_WRITER_SRC_CL_ICLTILE_H
#define COMPUTE_KERNEL_WRITER_SRC_CL_ICLTILE_H

#include "src/ITile.h"

namespace ckw
{
// Forward declarations
class TileInfo;

/** Interface for the OpenCL specific tile */
class ICLTile : public ITile,                              // classes inherited
                public IVectorAccess, public IScalarAccess // interfaces implemented
{
public:
    // Inherited method overridden
    std::vector<int32_t> supported_vector_lengths() const override;

protected:
    void validate_tile_info(const TileInfo &info) const;
};
} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_SRC_CL_ICLTILE_H */
