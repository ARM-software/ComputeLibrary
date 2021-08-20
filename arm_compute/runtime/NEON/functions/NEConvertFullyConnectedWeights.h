/*
 * Copyright (c) 2018-2021 Arm Limited.
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

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to run @ref cpu::kernels::CpuConvertFullyConnectedWeightsKernel. */
class NEConvertFullyConnectedWeights : public IFunction
{
public:
    /** Default constructor */
    NEConvertFullyConnectedWeights();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvertFullyConnectedWeights(const NEConvertFullyConnectedWeights &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvertFullyConnectedWeights &operator=(const NEConvertFullyConnectedWeights &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConvertFullyConnectedWeights(NEConvertFullyConnectedWeights &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConvertFullyConnectedWeights &operator=(NEConvertFullyConnectedWeights &&) = delete;
    /** Default destructor */
    ~NEConvertFullyConnectedWeights();
    /** Initialize the function.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
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
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NECONVERTFULLYCONNECTEDWEIGHTS_H */
