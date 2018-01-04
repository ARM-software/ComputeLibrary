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
#ifndef __ARM_COMPUTE_GCWEIGHTSRESHAPEKERNEL_H__
#define __ARM_COMPUTE_GCWEIGHTSRESHAPEKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class GCWeightsReshapeKernel : public IGCKernel
{
public:
    /** Constructor.*/
    GCWeightsReshapeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCWeightsReshapeKernel(const GCWeightsReshapeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCWeightsReshapeKernel &operator=(const GCWeightsReshapeKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCWeightsReshapeKernel(GCWeightsReshapeKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCWeightsReshapeKernel &operator=(GCWeightsReshapeKernel &&) = default;
    /** Default destructor */
    ~GCWeightsReshapeKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input tensor to convert. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM] if shared,
     *                    and 5D tensor with dimensions [kernel_x, kernel_y, IFM, OFM,  batches] if unshared. Data types supported: F16, F32
     * @param[in]  biases The shared biases tensor to append.  Bias is 1D tensor with dimensions [OFM] if shared and 2D tensor with
     *                    dimensions [OFM, batches] if unshared. Data types supported: Same as @p input
     *                    @warning Appending biases to weights reshaped matrix is not supported for quantized asymmetric types.
     * @param[out] output The output tensor. Should be a 2D Tensor. Data types supported: Same as @p input
     */
    void configure(const IGCTensor *input, const IGCTensor *biases, IGCTensor *output);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    const IGCTensor *_biases;
    IGCTensor       *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_GCWEIGHTSRESHAPEKERNEL_H__ */