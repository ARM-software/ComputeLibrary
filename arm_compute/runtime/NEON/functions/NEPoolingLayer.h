/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEPOOLINGLAYER_H__
#define __ARM_COMPUTE_NEPOOLINGLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEPoolingLayerKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Basic function to simulate a pooling layer with the specified pooling operation. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel (executed if padding size is different from zero)
 * -# @ref NEPoolingLayerKernel
 */
class NEPoolingLayer : public IFunction
{
public:
    /** Constructor */
    NEPoolingLayer();
    /** Set the input and output tensors.
     *
     * @note QS8, QS16 and F16 are supported for pool sizes 2 and 3 only
     *
     * @param[in, out] input     Source tensor. (Written to only when padding != 0) Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[out]     output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]      pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     */
    void configure(ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPoolingLayer
     *
     * @note QS8, QS16 and F16 are supported for pool sizes 2 and 3 only
     *
     * @param[in] input     Source tensor. (Written to only when padding != 0) Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[in] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run() override;

private:
    NEPoolingLayerKernel _pooling_layer_kernel;
    NEFillBorderKernel   _border_handler;
    bool                 _is_global_pooling_layer;
};
}
#endif /* __ARM_COMPUTE_NEPOOLINGLAYER_H__ */
