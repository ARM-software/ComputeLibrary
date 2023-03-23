/*
 * Copyright (c) 2019-2020, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_NEBATCHTOSPACELAYERKERNEL_H
#define ARM_COMPUTE_NEBATCHTOSPACELAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the batch to space kernel */
class NEBatchToSpaceLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEBatchToSpaceLayerKernel";
    }
    /** Default constructor */
    NEBatchToSpaceLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchToSpaceLayerKernel(const NEBatchToSpaceLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBatchToSpaceLayerKernel &operator=(const NEBatchToSpaceLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEBatchToSpaceLayerKernel(NEBatchToSpaceLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEBatchToSpaceLayerKernel &operator=(NEBatchToSpaceLayerKernel &&) = default;
    /** Default destructor */
    ~NEBatchToSpaceLayerKernel() = default;
    /** Initialise the kernel's inputs and output.
     *
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape 1-D tensor with shape [M]. Data types supported: S32
     * @param[out] output      Tensor output. Data types supported: same as @p input
     *
     * @deprecated This method for dynamic block shape is not fully mature and will be removed in 23.08 release
     */
    void configure(const ITensor *input, const ITensor *block_shape, ITensor *output);
    /** Initialise the kernel's inputs and output (Static block shape).
     *
     * @param[in]  input         Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[out] output        Tensor output. Data types supported: same as @p input
     * @param[in]  crop_info     Specifies how the output shape is cropped after batch to space is performed
     */
    void configure(const ITensor *input, int32_t block_shape_x, int32_t block_shape_y, ITensor *output, const CropInfo &crop_info = CropInfo{});
    /** Static function to check if given info will lead to a valid configuration of @ref NEBatchToSpaceLayerKernel
     *
     * @param[in] input       Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in] block_shape 1-D tensor with shape [M]. Data types supported: S32
     * @param[in] output      Tensor output. Data types supported: same as @p input
     *
     * @return a status
     *
     * @deprecated This method for dynamic block shape is not fully mature and will be removed in 23.08 release
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEBatchToSpaceLayerKernel (Static block shape).
     *
     * @param[in] input         Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in] block_shape_x Block shape x value.
     * @param[in] block_shape_y Block shape y value.
     * @param[in] output        Tensor output. Data types supported: same as @p input
     * @param[in] crop_info     Specifies how the output shape is cropped after batch to space is performed
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, int32_t block_shape_x, int32_t block_shape_y, const ITensorInfo *output, const CropInfo &crop_info = CropInfo{});

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;       /**< Source tensor */
    const ITensor *_block_shape; /**< Block shape tensor */
    ITensor       *_output;      /**< Destination tensor */
    DataLayout     _data_layout; /**< Data layout to  be used at run-time */
    int32_t        _block_shape_x;
    int32_t        _block_shape_y;
    CropInfo       _crop_info; /**< Information related to cropping performed on output after the operation */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEBATCHTOSPACELAYERKERNEL_H */
