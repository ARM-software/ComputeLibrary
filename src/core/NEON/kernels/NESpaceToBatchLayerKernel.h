/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NESPACETOBATCHLAYERKERNEL_H
#define ARM_COMPUTE_NESPACETOBATCHLAYERKERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declaration
class ITensor;

/** Interface for the space to batch kernel */
class NESpaceToBatchLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NESpaceToBatchLayerKernel";
    }
    /** Default constructor */
    NESpaceToBatchLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESpaceToBatchLayerKernel(const NESpaceToBatchLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NESpaceToBatchLayerKernel &operator=(const NESpaceToBatchLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NESpaceToBatchLayerKernel(NESpaceToBatchLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NESpaceToBatchLayerKernel &operator=(NESpaceToBatchLayerKernel &&) = default;
    /** Default destructor */
    ~NESpaceToBatchLayerKernel() = default;
    /** Initialise the kernel's inputs and output.
     *
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape 1-D tensor with shape [M]. Supported M: 2. Data types supported: S32
     * @param[in]  paddings    2-D tensor with shape [2, M] (First dimension is the fastest-changing dimension). Supported M: 2. Data types supported: S32
     * @param[out] output      Tensor output. Data types supported: same as @p input
     */
    void configure(const ITensor *input, const ITensor *block_shape, const ITensor *paddings, ITensor *output);
    /** Initialise the kernel's input and output. (Static block shape and paddings)
     *
     * @param[in]  input         Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[in]  padding_left  The padding at the beginning of every dimension of the output tensor.
     * @param[in]  padding_right The padding at the end of every dimension of the output tensor.
     * @param[out] output        Tensor output. Data types supported: same as @p input
     */
    void configure(const ITensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NESpaceToBatchLayerKernel
     *
     * @param[in] input       Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in] block_shape 1-D tensor with shape [M]. Supported M: 2. Data types supported: S32
     * @param[in] paddings    2-D tensor with shape [2, M] (First dimension is the fastest-changing dimension). Supported M: 2. Data types supported: S32
     * @param[in] output      Tensor output. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *paddings, const ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NESpaceToBatchLayerKernel (Static block shape and paddings)
     *
     * @param[in] input         Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[in] block_shape_x Block shape x value.
     * @param[in] block_shape_y Block shape y value.
     * @param[in] padding_left  The padding at the beginning of every dimension of the output tensor.
     * @param[in] padding_right The padding at the end of every dimension of the output tensor.
     * @param[in] output        Tensor output. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;       /**< Source tensor */
    const ITensor *_block_shape; /**< Block shape tensor for dynamic evaluation */
    const ITensor *_paddings;    /**< Paddings tensor for dynamic evaluation */
    ITensor       *_output;      /**< Destination tensor */
    DataLayout     _data_layout; /**< Data layout to be used at run-time */

    Size2D _padding_left;
    int    _block_shape_x;
    int    _block_shape_y;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NESPACETOBATCHLAYERKERNEL_H */
