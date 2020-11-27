/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GCPOOLINGLAYER_H
#define ARM_COMPUTE_GCPOOLINGLAYER_H

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCFillBorderKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCPoolingLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCTensorShiftKernel.h"
#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to simulate a pooling layer with the specified pooling operation. This function calls the following OpenGL ES kernels:
 *
 * -# @ref GCFillBorderKernel (executed if padding size is different from zero)
 * -# @ref GCPoolingLayerKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCPoolingLayer : public IFunction
{
public:
    GCPoolingLayer();
    /** Set the input and output tensors.
     *
     * @param[in,out] input     Source tensor. (Written to only when padding != 0) Data types supported: F16/F32.
     * @param[out]    output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]     pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out]    indices   (optional) The indices of the maximal values. Data type supported: U32.
     */
    void configure(IGCTensor *input, IGCTensor *output, const PoolingLayerInfo &pool_info, IGCTensor *indices = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref GCPoolingLayer
     *
     * @param[in] input     Source tensor info. Data types supported: F16/F32.
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[in] indices   (optional) The indices of the maximal values. Data type supported: U32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info, const ITensorInfo *indices = nullptr);

    void run() override final;

private:
    std::unique_ptr<IGCKernel> _kernel;
    GCFillBorderKernel         _border_handler;
    GCTensorShiftKernel        _shift_handler;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_GCPOOLINGLAYER_H */
