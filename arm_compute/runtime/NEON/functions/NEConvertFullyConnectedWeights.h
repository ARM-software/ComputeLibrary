/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_NECONVERTFULLYCONNECTEDWEIGHTS_H
#define ARM_COMPUTE_NECONVERTFULLYCONNECTEDWEIGHTS_H

#include "arm_compute/core/NEON/kernels/NEConvertFullyConnectedWeightsKernel.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/ITransformWeights.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to run @ref NEConvertFullyConnectedWeightsKernel. */
class NEConvertFullyConnectedWeights : public IFunction
{
public:
    /** Default constructor */
    NEConvertFullyConnectedWeights();
    /** Initialize the function.
     *
     * @param[in]  input                Source weights tensor to convert. Must be 2 dimensional. Data types supported: All.
     * @param[out] output               The converted weights tensor. Shape and Data Type: Same as @p input.
     * @param[in]  original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in]  data_layout          The data layout the weights have been trained in.
     */
    void configure(const ITensor *input, ITensor *output, const TensorShape &original_input_shape, DataLayout data_layout);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConvertFullyConnectedWeights
     *
     * @param[in] input                Source weights tensor info to convert. Must be 2 dimensional. Data types supported: All.
     * @param[in] output               The converted weights tensor info. Shape and Data Type: Same as @p input.
     * @param[in] original_input_shape Shape of the original input tensor (the one entering fully connected layer).
     * @param[in] data_layout          The data layout the weights have been trained in.
     *
     * @return A Status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const TensorShape &original_input_shape, DataLayout data_layout);

    // Inherited methods overriden:
    void run() override;

private:
    NEConvertFullyConnectedWeightsKernel _kernel;
};

namespace weights_transformations
{
/** Basic function to run @ref NEConvertFullyConnectedWeightsKernel. */
class NEConvertFullyConnectedWeightsManaged : public ITransformWeights
{
public:
    void run() override
    {
        _output.allocator()->allocate();
        _func.run();
        _reshape_run = true;
    }

    void release() override
    {
        _output.allocator()->free();
    }

    ITensor *get_weights() override
    {
        return &_output;
    }

    uint32_t uid() override
    {
        return _uid;
    }

    void configure(const ITensor *input, const TensorShape &original_input_shape, DataLayout data_layout)
    {
        _func.configure(input, &_output, original_input_shape, data_layout);
    }

private:
    static constexpr uint32_t      _uid = 0x4;
    Tensor                         _output{};
    NEConvertFullyConnectedWeights _func{};
};
} // namespace weights_transformations
} // namespace arm_compute
#endif /* ARM_COMPUTE_NECONVERTFULLYCONNECTEDWEIGHTS_H */
