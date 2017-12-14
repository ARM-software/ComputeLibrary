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

#ifndef __ARM_COMPUTE_NEDEPTHCONCATENATEKERNEL_H__
#define __ARM_COMPUTE_NEDEPTHCONCATENATEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the depth concatenate kernel.
 *  The input tensor will be concatenated into the output tensor.
 */
class NEDepthConcatenateLayerKernel : public INEKernel
{
public:
    /** Default constructor */
    NEDepthConcatenateLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthConcatenateLayerKernel(const NEDepthConcatenateLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthConcatenateLayerKernel &operator=(const NEDepthConcatenateLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDepthConcatenateLayerKernel(NEDepthConcatenateLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDepthConcatenateLayerKernel &operator=(NEDepthConcatenateLayerKernel &&) = default;
    /** Default destructor */
    ~NEDepthConcatenateLayerKernel() = default;
    /** Initialise the kernel's inputs and output
     *
     * @param[in]     input        Input tensor. Data types supported: QS8/QS16/F16/F32.
     * @param[in]     depth_offset The offset on the Z axis.
     * @param[in,out] output       Output tensor. Data types supported: Same as @p input.
     *
     * @note: The output tensor's low two dimensions can't be smaller than the input one's.
     * @note: The gaps between the two lowest dimensions of input and output need to be divisible by 2.
     *
     */
    void configure(const ITensor *input, unsigned int depth_offset, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    using DepthConcatFunction = void(const ITensor *in, ITensor *out, std::pair<int, int> start_xy, int depth_offset, const Window &window);

private:
    DepthConcatFunction *_func;
    const ITensor       *_input;
    ITensor             *_output;
    int                  _top_bottom;
    int                  _left_right;
    unsigned int         _depth_offset;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEDEPTHCONCATENATEKERNEL_H__ */
