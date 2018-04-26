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
#ifndef __ARM_COMPUTE_NECONVERTFULLYCONNECTEDWEIGHTSKERNEL_H__
#define __ARM_COMPUTE_NECONVERTFULLYCONNECTEDWEIGHTSKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface to convert the 2D Fully Connected weights from NCHW to NHWC or vice versa.
 *
 * @note This function can be applied to the 2D weights used by a Fully Connected layer if:
 *       - It follows a Convolution layer
 *       - The data layout used by the network does not match the one the model has been trained in.
 *
 * @note This function assumes the weights are already reshaped (transposed)
 */
class NEConvertFullyConnectedWeightsKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEConvertFullyConnectedWeightsKernel";
    }
    /** Default constructor */
    NEConvertFullyConnectedWeightsKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvertFullyConnectedWeightsKernel(const NEConvertFullyConnectedWeightsKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvertFullyConnectedWeightsKernel &operator=(const NEConvertFullyConnectedWeightsKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEConvertFullyConnectedWeightsKernel(NEConvertFullyConnectedWeightsKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEConvertFullyConnectedWeightsKernel &operator=(NEConvertFullyConnectedWeightsKernel &&) = default;
    /** Default destructor */
    ~NEConvertFullyConnectedWeightsKernel() = default;
    /** Set the input and output tensor.
     *
     * @param[in]  input                Source weights tensor to convert. Must be 2 dimensional. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/U32/S32/QS32/F16/F32.
     * @param[out] output               The converted weights tensor. Shape and Data Type: Same as @p input.
     * @param[in]  original_input_shape Shape of the original input tensor (the one entering fully connected layer). Must be in NCHW format.
     * @param[in]  data_layout          The data layout the weights have been trained in.
     */
    void configure(const ITensor *input, ITensor *output, const TensorShape &original_input_shape, DataLayout data_layout);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConvertFullyConnectedWeightsKernel
     *
     * @param[in] input                Source weights tensor info to convert. Must be 2 dimensional. Data types supported: U8/S8/QS8/QASYMM8/U16/S16/QS16/U32/S32/QS32/F16/F32.
     * @param[in] output               The converted weights tensor info. Shape and Data Type: Same as @p input.
     * @param[in] original_input_shape Shape of the original input tensor (the one entering fully connected layer). Must be in NCHW format.
     * @param[in] data_layout          The data layout the weights have been trained in.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const TensorShape &original_input_shape, DataLayout data_layout);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run the permute
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T>
    void run_convert_fc_weights(const Window &window);

    const ITensor *_input;
    ITensor       *_output;
    unsigned int   _factor1; /*  equals to the number of elements per original input plane if @p data_layout == NCHW; its number of channels otherwise */
    unsigned int   _factor2; /*  equals to the number of elements per original input plane if @p data_layout == NHWC; its number of channels otherwise */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NECONVERTFULLYCONNECTEDWEIGHTSKERNEL_H__ */
