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
#ifndef __ARM_COMPUTE_CLSPACETOBATCHLAYER_H__
#define __ARM_COMPUTE_CLSPACETOBATCHLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLMemsetKernel.h"
#include "arm_compute/core/CL/kernels/CLSpaceToBatchLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to spatial divide a tensor. This function calls the following OpenCL kernels/functions:
 *
 *  -# @ref CLMemsetKernel
 *  -# @ref CLSpaceToBatchLayerKernel
 */
class CLSpaceToBatchLayer : public IFunction
{
public:
    /** Default constructor */
    CLSpaceToBatchLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSpaceToBatchLayer(const CLSpaceToBatchLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLSpaceToBatchLayer &operator=(const CLSpaceToBatchLayer &) = delete;
    /** Allow instances of this class to be moved */
    CLSpaceToBatchLayer(CLSpaceToBatchLayer &&) = default;
    /** Allow instances of this class to be moved */
    CLSpaceToBatchLayer &operator=(CLSpaceToBatchLayer &&) = default;
    /** Default destructor */
    virtual ~CLSpaceToBatchLayer() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape 1-D tensor with shape [M]. Data types supported: S32
     * @param[in]  paddings    2-D tensor with shape [2, M]. Data types supported: S32
     * @param[out] output      Tensor output. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, const ICLTensor *block_shape, const ICLTensor *paddings, ICLTensor *output);
    /** Set the input and output tensors. (Static block shape and paddings)
     *
     * @param[in]  input         Tensor input. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[in]  padding_left  The left padding of the output tensor.
     * @param[in]  padding_right The right padding of the output tensor.
     * @param[out] output        Tensor output. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSpaceToBatchLayer
     *
     * @param[in]  input       Tensor input info. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape block shape tensor info with shape [M]. Data types supported: S32
     * @param[in]  paddings    paddings tensor info with shape [2, M]. Data types supported: S32
     * @param[out] output      Tensor output info. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *paddings, const ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSpaceToBatchLayer (Static block shape and paddings)
     *
     * @param[in]  input         Tensor input info. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[in]  padding_left  The left padding of the output tensor.
     * @param[in]  padding_right The right padding of the output tensor.
     * @param[out] output        Tensor output info. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const int block_shape_x, const int block_shape_y, const Size2D &padding_left, const Size2D &padding_right, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    CLSpaceToBatchLayerKernel _space_to_batch_kernel; /**< SpaceToBatch kernel to run */
    CLMemsetKernel            _memset_kernel;         /**< Memset kernel to run */
    bool                      _has_padding;           /**< Flag to check if the output has padding */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLSPACETOBATCHLAYER_H__ */
