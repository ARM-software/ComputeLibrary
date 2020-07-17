/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEMAXUNPOOLINGLAYER_H
#define ARM_COMPUTE_NEMAXUNPOOLINGLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEMaxUnpoolingLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEMemsetKernel.h"

namespace arm_compute
{
class ITensor;

/** Function to perform MaxUnpooling. This function calls the following NEON kernels:
 *
 * -# @ref NEMemsetKernel
 * -# @ref NEMaxUnpoolingLayerKernel
 */
class NEMaxUnpoolingLayer : public IFunction
{
public:
    /** Constructor */
    NEMaxUnpoolingLayer();
    /** Set the input and output tensors.
     *
     * @note Only supported pool size 2
     *
     * @param[in, out] input     Source tensor. (Written to only when padding != 0) Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out]     output    Destination tensor. Data types supported: Same as @p input.
     * @param[out]     indices   The indices of the maximal values. Data type supported: U32.
     * @param[in]      pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     */
    void configure(ITensor *input, ITensor *indices, ITensor *output, const PoolingLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEMaxUnpoolingLayer
     *
     * @note Only supported pool size 2
     *
     * @param[in] input     Source tensor. (Written to only when padding != 0) Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] indices   The indices of the maximal values. Data type supported: U32.
     * @param[in] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, const PoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run() override;

private:
    NEMemsetKernel            _memset_kernel;
    NEMaxUnpoolingLayerKernel _unpooling_layer_kernel;
};
}
#endif /* ARM_COMPUTE_NEMAXUNPOOLINGLAYER_H */
