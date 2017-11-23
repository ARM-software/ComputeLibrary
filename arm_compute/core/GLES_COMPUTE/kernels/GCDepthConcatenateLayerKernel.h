/*
 * Copyright (c) 2017 ARM Limited.
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

#ifndef __ARM_COMPUTE_GCDEPTHCONCATENATEKERNEL_H__
#define __ARM_COMPUTE_GCDEPTHCONCATENATEKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the depth concatenate kernel.
 *  The input tensor will be concatenated into the output tensor.
 */
class GCDepthConcatenateLayerKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCDepthConcatenateLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCDepthConcatenateLayerKernel(const GCDepthConcatenateLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCDepthConcatenateLayerKernel &operator=(const GCDepthConcatenateLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCDepthConcatenateLayerKernel(GCDepthConcatenateLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCDepthConcatenateLayerKernel &operator=(GCDepthConcatenateLayerKernel &&) = default;
    /** Default destructor */
    ~GCDepthConcatenateLayerKernel() = default;
    /** Initialise the kernel's inputs and output
     *
     * @param[in]     input        Input tensor. Data types supported: F16/F32.
     * @param[in]     depth_offset The offset on the Z axis.
     * @param[in,out] output       Output tensor. Data types supported: Same as @p input.
     *
     * @note: The output tensor's low two dimensions can't be smaller than the input one's.
     * @note: The gaps between the two lowest dimensions of input and output need to be divisible by 2.
     *
     */
    void configure(const IGCTensor *input, unsigned int depth_offset, IGCTensor *output);

    // Inherited methods overridden:
    void run(const Window &window) override;
    BorderSize border_size() const override;

private:
    const IGCTensor *_input;
    IGCTensor       *_output;
    int              _top_bottom;
    int              _left_right;
};
}
#endif /* __ARM_COMPUTE_GCDEPTHCONCATENATEKERNEL_H__ */
