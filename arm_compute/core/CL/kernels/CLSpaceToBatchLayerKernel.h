/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLSPACETOBATCHLAYERKERNEL_H__
#define __ARM_COMPUTE_CLSPACETOBATCHLAYERKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the space to batch kernel */
class CLSpaceToBatchLayerKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLSpaceToBatchLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSpaceToBatchLayerKernel(const CLSpaceToBatchLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSpaceToBatchLayerKernel &operator=(const CLSpaceToBatchLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLSpaceToBatchLayerKernel(CLSpaceToBatchLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLSpaceToBatchLayerKernel &operator=(CLSpaceToBatchLayerKernel &&) = default;
    /** Default destructor */
    ~CLSpaceToBatchLayerKernel() = default;
    /** Initialise the kernel's inputs and output.
     *
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape 1-D tensor with shape [M]. Data types supported: S32
     * @param[in]  paddings    2-D tensor with shape [2, M]. Data types supported: S32
     * @param[out] output      Tensor output. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, const ICLTensor *block_shape, const ICLTensor *paddings, ICLTensor *output);
    /** Initialise the kernel's input and output. (Static block shape and paddings)
     *
     * @param[in]  input         Tensor input. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[in]  padding_left  The left padding of the output tensor.
     * @param[in]  padding_right The right padding of the output tensor.
     * @param[out] output        Tensor output. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSpaceToBatchLayerKernel
     *
     * @param[in] input       Tensor input. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] block_shape 1-D tensor with shape [M]. Data types supported: S32
     * @param[in] paddings    2-D tensor with shape [2, M]. Data types supported: S32
     * @param[in] output      Tensor output. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *paddings, const ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSpaceToBatchLayerKernel (Static block shape and paddings)
     *
     * @param[in] input         Tensor input. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in] block_shape_x Block shape x value.
     * @param[in] block_shape_y Block shape y value.
     * @param[in] padding_left  The left padding of the output tensor.
     * @param[in] padding_right The right padding of the output tensor.
     * @param[in] output        Tensor output. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;       /**< Source tensor */
    const ICLTensor *_block_shape; /**< Block shape tensor */
    const ICLTensor *_paddings;    /**< Paddings tensor */
    ICLTensor       *_output;      /**< Destination tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLSPACETOBATCHLAYERKERNEL_H__ */
