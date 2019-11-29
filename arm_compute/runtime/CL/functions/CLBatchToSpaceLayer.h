/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_CLBATCHTOSPACELAYER_H
#define ARM_COMPUTE_CLBATCHTOSPACELAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLBatchToSpaceLayerKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLBatchToSpaceLayerKernel. */
class CLBatchToSpaceLayer : public IFunction
{
public:
    /** Default constructor */
    CLBatchToSpaceLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape 1-D tensor with shape [M]. Data types supported: S32
     * @param[out] output      Tensor output. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, const ICLTensor *block_shape, ICLTensor *output);
    /** Set the input and output tensors. (Static block shape).
     *
     * @param[in]  input         Tensor input. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[out] output        Tensor output. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, int32_t block_shape_x, int32_t block_shape_y, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLBatchToSpaceLayer
     *
     * @param[in]  input       Tensor input info. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape block shape tensor info with shape [M]. Data types supported: S32
     * @param[out] output      Tensor output info. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLBatchToSpaceLayer (Static block shape).
     *
     * @param[in]  input         Tensor input info. Supported tensor rank: 4. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     * @param[in]  block_shape_x Block shape x value.
     * @param[in]  block_shape_y Block shape y value.
     * @param[out] output        Tensor output info. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, int32_t block_shape_x, int32_t block_shape_y, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    CLBatchToSpaceLayerKernel _batch_to_space_kernel; /**< CLBatchToSpaceLayerKernel to run */
};
}
#endif /* ARM_COMPUTE_CLBATCHTOSPACELAYER_H */
