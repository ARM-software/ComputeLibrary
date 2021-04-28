/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEDEPTHTOSPACELAYER_H
#define ARM_COMPUTE_NEDEPTHTOSPACELAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to run @ref NEDepthToSpaceLayerKernel. */
class NEDepthToSpaceLayer : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEDepthToSpaceLayer() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthToSpaceLayer(const NEDepthToSpaceLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthToSpaceLayer &operator=(const NEDepthToSpaceLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEDepthToSpaceLayer(NEDepthToSpaceLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEDepthToSpaceLayer &operator=(NEDepthToSpaceLayer &&) = delete;
    /** Default destructor */
    ~NEDepthToSpaceLayer() = default;
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: All
     * @param[out] output      Tensor output. Data types supported: same as @p input
     * @param[in]  block_shape Block shape value.
     */
    void configure(const ITensor *input, ITensor *output, int32_t block_shape);
    /** Static function to check if given info will lead to a valid configuration of @ref NEDepthToSpaceLayer.
     *
     * @param[in] input       Tensor input info. Supported tensor rank: 4. Data types supported: All
     * @param[in] output      Tensor output info. Data types supported: same as @p input
     * @param[in] block_shape Block shape x value.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEDEPTHTOSPACELAYER_H */
