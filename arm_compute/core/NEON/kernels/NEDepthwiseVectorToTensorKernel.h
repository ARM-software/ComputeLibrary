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
#ifndef __ARM_COMPUTE_NEDEPTHWISEVECTORTOTENSORKERNEL_H__
#define __ARM_COMPUTE_NEDEPTHWISEVECTORTOTENSORKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the depthwise vector to tensor kernel.
 *
 *  This kernel takes the 1D tensor that's been produced by the MatrixVectorMultiply
 *  kernel and reshapes it to given width and height (previously calculated, based
 *  on input/weights dimensions and convolution strides and padding).
 *
 **/
class NEDepthwiseVectorToTensorKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDepthwiseVectorToTensorKernel";
    }
    /** Default constructor */
    NEDepthwiseVectorToTensorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseVectorToTensorKernel(const NEDepthwiseVectorToTensorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseVectorToTensorKernel &operator=(const NEDepthwiseVectorToTensorKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDepthwiseVectorToTensorKernel(NEDepthwiseVectorToTensorKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDepthwiseVectorToTensorKernel &operator=(NEDepthwiseVectorToTensorKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input vector to convert. Data type supported: QASYMM8/S32/F16/F32.
     * @param[out] output The output tensor. 3 lower dimensions represent a single input [width, height, IFM]. Data type supported: same as @p input.
     * @param[in]  conv_w The converted tensor's width.
     * @param[in]  conv_h The converted tensor's height.
     */
    void configure(const ITensor *input, ITensor *output, size_t conv_w, size_t conv_h);
    /** Static function to check if given info will lead to a valid configuration of @ref NEDepthwiseVectorToTensorKernel
     *
     * @param[in] input  The input vector to convert. Data type supported: QASYMM8/S32/F16/F32.
     * @param[in] output The output tensor. 3 lower dimensions represent a single input [width, height, IFM]. Data type supported: same as @p input.
     * @param[in] conv_w The converted tensor's width.
     * @param[in] conv_h The converted tensor's height.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, size_t conv_w, size_t conv_h);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run the vector to tensor reshape used for the depthwise convolution layer case
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T>
    void vector_to_tensor(const Window &window);
    /** Common signature for all the specialised depthwise vector to tensor functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using DepthwiseVectorToTensorFunctionPtr = void (NEDepthwiseVectorToTensorKernel::*)(const Window &window);

private:
    DepthwiseVectorToTensorFunctionPtr _func;
    const ITensor                     *_input;
    ITensor                           *_output;
    std::pair<size_t, size_t> _conv_dims;
};
} // arm_compute
#endif /*__ARM_COMPUTE_NEDEPTHWISEVECTORTOTENSORKERNEL_H__ */
