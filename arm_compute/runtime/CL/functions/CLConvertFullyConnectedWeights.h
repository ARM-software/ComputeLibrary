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
#ifndef ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTS_H
#define ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTS_H

#include "arm_compute/core/CL/kernels/CLConvertFullyConnectedWeightsKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/ITransformWeights.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLConvertFullyConnectedWeightsKernel. */
class CLConvertFullyConnectedWeights : public ICLSimpleFunction
{
public:
    /** Initialize the function.
     *
     * @param[in]  input                Source weights tensor to convert. Must be 2 dimensional. Data types supported: All.
     * @param[out] output               The converted weights tensor. Shape and Data Type: Same as @p input.
     * @param[in]  original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in]  data_layout          The data layout the weights have been trained in.
     *
     * @return A status
     */
    void configure(const ICLTensor *input, ICLTensor *output, const TensorShape &original_input_shape, DataLayout data_layout);
    /** Initialize the function.
     *
     * @param[in]  compile_context      The compile context to be used.
     * @param[in]  input                Source weights tensor to convert. Must be 2 dimensional. Data types supported: All.
     * @param[out] output               The converted weights tensor. Shape and Data Type: Same as @p input.
     * @param[in]  original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in]  data_layout          The data layout the weights have been trained in.
     *
     * @return A status
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, const TensorShape &original_input_shape, DataLayout data_layout);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConvertFullyConnectedWeights
     *
     * @param[in] input                Source weights tensor info to convert. Must be 2 dimensional. Data types supported: All.
     * @param[in] output               The converted weights tensor info. Shape and Data Type: Same as @p input.
     * @param[in] original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in] data_layout          The data layout the weights have been trained in.
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const TensorShape &original_input_shape, DataLayout data_layout);
};

namespace weights_transformations
{
/** Basic function to run @ref CLConvertFullyConnectedWeightsKernel. */
class CLConvertFullyConnectedWeightsManaged : public ITransformWeights
{
public:
    //Inherited method override
    void run() override
    {
        _output.allocator()->allocate();
        _func.run();
        _reshape_run = true;
    }

    //Inherited method override
    void release() override
    {
        _output.allocator()->free();
    }

    //Inherited method override
    ICLTensor *get_weights() override
    {
        return &_output;
    }

    //Inherited method override
    uint32_t uid() override
    {
        return _uid;
    }
    /** Configures the @ref CLConvertFullyConnectedWeights function
     *
     * @param[in] input                Source weights tensor info to convert.  Data type supported: All.
     * @param[in] original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in] data_layout          The data layout the weights have been trained in.
     */
    void configure(const ICLTensor *input, const TensorShape &original_input_shape, DataLayout data_layout)
    {
        configure(CLKernelLibrary::get().get_compile_context(), input, original_input_shape, data_layout);
    }
    /** Configures the @ref CLConvertFullyConnectedWeights function
     *
     * @param[in] compile_context      The compile context to be used.
     * @param[in] input                Source weights tensor info to convert.  Data type supported: All.
     * @param[in] original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in] data_layout          The data layout the weights have been trained in.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const TensorShape &original_input_shape, DataLayout data_layout)
    {
        _func.configure(compile_context, input, &_output, original_input_shape, data_layout);
    }

private:
    static constexpr uint32_t      _uid = 0x5;
    CLTensor                       _output{};
    CLConvertFullyConnectedWeights _func{};
};
} // namespace weights_transformations
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLCONVERTFULLYCONNECTEDWEIGHTS_H */
