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

#ifndef CKW_SRC_TENSOR3DMAPPER_H
#define CKW_SRC_TENSOR3DMAPPER_H

#include <string>

namespace ckw
{
// Forward declarations
class ITensor;
enum class TensorSamplerFormat;
struct TileVariable;

/** This internal-only class is responsible to map an Nd tensor spatial dimensions to a 3d tensor spatial dimensions with the
 *  help of TensorSamplerFormat.
 *  Attention: The batch is not considered as a spatial dimension and it is treated as an offset
 *
 *  The aim of the dimensionality reduction is primarily to reduce
 *  the address calculation to:
 *      x + y * stride_y + z * stride_z + offset, where offset is determined by the batch (for example, b * stride_batch).
 *
 */
class Tensor3dMapper
{
public:
    /** Constructor */
    Tensor3dMapper(ITensor *tensor, TensorSamplerFormat format);

    /** Get dimension x as string */
    TileVariable dim_x() const;

    /** Get dimension y as string */
    TileVariable dim_y() const;

    /** Get dimension z as string */
    TileVariable dim_z() const;

    /** Get batch dimension as string */
    TileVariable dim_batch() const;

    /** Get stride for dimension x as string */
    TileVariable stride_x() const;

    /** Get stride for dimension y as string */
    TileVariable stride_y() const;

    /** Get stride for dimension z as string */
    TileVariable stride_z() const;

    /** Get stride for batch dimension as string */
    TileVariable stride_batch() const;

private:
    ITensor            *_tensor;
    TensorSamplerFormat _format;
};
} // namespace ckw

#endif /* CKW_SRC_TENSOR3DMAPPER_H */
