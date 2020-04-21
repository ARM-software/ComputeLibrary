/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTSKERNEL_H
#define ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTSKERNEL_H

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface to convert the 2D Fully Connected weights from NCHW to NHWC or vice versa.
 *
 * @note This function can be applied to the 2D weights used by a Fully Connected layer if:
 *       - It follows a Convolution layer
 *       - The data layout used by the network does not match the one the model has been trained in.
 *
 * @note This function assumes the weights are already reshaped (transposed)
 */
class CLConvertFullyConnectedWeightsKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLConvertFullyConnectedWeightsKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConvertFullyConnectedWeightsKernel(const CLConvertFullyConnectedWeightsKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConvertFullyConnectedWeightsKernel &operator=(const CLConvertFullyConnectedWeightsKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLConvertFullyConnectedWeightsKernel(CLConvertFullyConnectedWeightsKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLConvertFullyConnectedWeightsKernel &operator=(CLConvertFullyConnectedWeightsKernel &&) = default;
    /** Default destructor */
    ~CLConvertFullyConnectedWeightsKernel() = default;
    /** Set the input and output tensor.
     *
     * @param[in]  input                Source weights tensor to convert. Must be 2 dimensional. Data types supported: All.
     * @param[out] output               The converted weights tensor. Shape and Data Type: Same as @p input.
     * @param[in]  original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in]  data_layout          The data layout the weights have been trained in.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const TensorShape &original_input_shape, DataLayout data_layout);
    /** Set the input and output tensor.
     *
     * @param[in]  compile_context      The compile context to be used.
     * @param[in]  input                Source weights tensor to convert. Must be 2 dimensional. Data types supported: All.
     * @param[out] output               The converted weights tensor. Shape and Data Type: Same as @p input.
     * @param[in]  original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in]  data_layout          The data layout the weights have been trained in.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const TensorShape &original_input_shape, DataLayout data_layout);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConvertFullyConnectedWeightsKernel
     *
     * @param[in] input                Source weights tensor info to convert. Must be 2 dimensional. Data types supported: All.
     * @param[in] output               The converted weights tensor info. Shape and Data Type: Same as @p input.
     * @param[in] original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in] data_layout          The data layout the weights have been trained in.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const TensorShape &original_input_shape, DataLayout data_layout);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTSKERNEL_H */
