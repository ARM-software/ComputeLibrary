/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CLBATCHTOSPACELAYERKERNEL_H
#define ARM_COMPUTE_CLBATCHTOSPACELAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the batch to space kernel */
class CLBatchToSpaceLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLBatchToSpaceLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLBatchToSpaceLayerKernel(const CLBatchToSpaceLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLBatchToSpaceLayerKernel &operator=(const CLBatchToSpaceLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLBatchToSpaceLayerKernel(CLBatchToSpaceLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLBatchToSpaceLayerKernel &operator=(CLBatchToSpaceLayerKernel &&) = default;
    /** Default destructor */
    ~CLBatchToSpaceLayerKernel() = default;
    /** Initialise the kernel's inputs and output.
     *
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape 1-D tensor with shape [M]. Data types supported: S32
     * @param[out] output      Tensor output. Data types supported: same as @p input
     *
     * @deprecated This method for dynamic block shape is not fully mature and will be removed in 23.08 release
     */
    void configure(const ICLTensor *input, const ICLTensor *block_shape, ICLTensor *output);
    /** Initialise the kernel's inputs and output.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape     1-D tensor with shape [M]. Data types supported: S32
     * @param[out] output          Tensor output. Data types supported: same as @p input
     *
     * @deprecated This method for dynamic block shape is not fully mature and will be removed in 23.08 release
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *block_shape, ICLTensor *output);
    /** Initialise the kernel's inputs and output (Static block shape).
     *
     * @param[in]  input         Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[out] output        Tensor output. Data types supported: same as @p input
     * @param[in]  crop_info     Specifies how the output shape is cropped after batch to space is performed
     */
    void configure(const ICLTensor *input, const int32_t block_shape_x, const int32_t block_shape_y, ICLTensor *output, const CropInfo &crop_info);
    /** Initialise the kernel's inputs and output (Static block shape).
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape_x   Block shape x value.
     * @param[in]  block_shape_y   Block shape y value.
     * @param[out] output          Tensor output. Data types supported: same as @p input
     * @param[in]  crop_info       Specifies how the output shape is cropped after batch to space is performed
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const int32_t block_shape_x, const int32_t block_shape_y, ICLTensor *output, const CropInfo &crop_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLBatchToSpaceLayerKernel
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
    /** Static function to check if given info will lead to a valid configuration of @ref CLBatchToSpaceLayerKernel (Static block shape).
     *
     * @param[in] input         Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in] block_shape_x Block shape x value.
     * @param[in] block_shape_y Block shape y value.
     * @param[in] output        Tensor output. Data types supported: same as @p input
     * @param[in] crop_info     Specifies how the output shape is cropped after batch to space is performed
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const int32_t block_shape_x, const int32_t block_shape_y, const ITensorInfo *output, const CropInfo &crop_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;       /**< Source tensor */
    const ICLTensor *_block_shape; /**< Block shape tensor */
    ICLTensor       *_output;      /**< Destination tensor */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLBATCHTOSPACELAYERKERNEL_H */
