/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_NEBATCHTOSPACELAYER_H
#define ARM_COMPUTE_NEBATCHTOSPACELAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref NEBatchToSpaceLayerKernel. */
class NEBatchToSpaceLayer : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEBatchToSpaceLayer() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchToSpaceLayer(const NEBatchToSpaceLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchToSpaceLayer &operator=(const NEBatchToSpaceLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEBatchToSpaceLayer(NEBatchToSpaceLayer &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEBatchToSpaceLayer &operator=(NEBatchToSpaceLayer &&) = delete;
    /** Default destructor */
    ~NEBatchToSpaceLayer() = default;
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0      |src1      |dst        |
     * |:---------|:---------|:----------|
     * |All       |s32       |All        |
     *
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape 1-D tensor with shape [M]. Data types supported: S32
     * @param[out] output      Tensor output. Data types supported: same as @p input
     *
     * @deprecated This method for dynamic block shape is not fully mature and will be removed in 23.08 release
     */
    ARM_COMPUTE_DEPRECATED_REL(23.05)
    void configure(const ITensor *input, const ITensor *block_shape, ITensor *output);
    /** Set the input and output tensors. (Static block shape).
     *
     * @param[in]  input         Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[out] output        Tensor output. Data types supported: same as @p input
     * @param[in]  crop_info     Specifies how the output shape is cropped after batch to space is performed
     */
    void configure(const ITensor *input, int32_t block_shape_x, int32_t block_shape_y, ITensor *output, const CropInfo &crop_info = CropInfo{});
    /** Static function to check if given info will lead to a valid configuration of @ref CLBatchToSpaceLayer
     *
     * @param[in]  input       Tensor input info. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape block shape tensor info with shape [M]. Data types supported: S32
     * @param[out] output      Tensor output info. Data types supported: same as @p input
     *
     * @return a status
     * @deprecated This method for dynamic block shape is not fully mature and will be removed in 23.08 release
     */
    ARM_COMPUTE_DEPRECATED_REL(23.05)
    static Status validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLBatchToSpaceLayer (Static block shape).
     *
     * @param[in]  input         Tensor input info. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[out] output        Tensor output info. Data types supported: same as @p input
     * @param[in]  crop_info     Specifies how the output shape is cropped after batch to space is performed
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, int32_t block_shape_x, int32_t block_shape_y, const ITensorInfo *output, const CropInfo &crop_info = CropInfo{});
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEBATCHTOSPACELAYER_H */
