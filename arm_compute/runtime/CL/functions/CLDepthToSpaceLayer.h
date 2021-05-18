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
#ifndef ARM_COMPUTE_CLDEPTHTOSPACELAYER_H
#define ARM_COMPUTE_CLDEPTHTOSPACELAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref CLDepthToSpaceLayerKernel. */
class CLDepthToSpaceLayer : public ICLSimpleFunction
{
public:
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
     * @param[in]  input       Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[out] output      Tensor output. Data types supported: same as @p input
     * @param[in]  block_shape Block shape value.
     */
    void configure(const ICLTensor *input, ICLTensor *output, int32_t block_shape);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Tensor input. Supported tensor rank: 4. Data types supported: All.
     * @param[out] output          Tensor output. Data types supported: same as @p input
     * @param[in]  block_shape     Block shape value.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, int32_t block_shape);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthToSpaceLayer.
     *
     * @param[in] input       Tensor input info. Supported tensor rank: 4. Data types supported: All.
     * @param[in] output      Tensor output info. Data types supported: same as @p input
     * @param[in] block_shape Block shape value.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape);
};
}
#endif /* ARM_COMPUTE_CLDEPTHTOSPACELAYER_H */
