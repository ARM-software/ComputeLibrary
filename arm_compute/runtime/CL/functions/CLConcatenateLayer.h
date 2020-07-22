/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCONCATENATELAYER_H
#define ARM_COMPUTE_CLCONCATENATELAYER_H

#include "arm_compute/runtime/CL/ICLOperator.h"
#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

#include <memory>
#include <vector>

namespace arm_compute
{
// Forward declarations
class ICLTensor;
class ITensorInfo;
class Status;

/** Basic function to execute concatenate tensors along a given axis. This function calls the following kernels:
 *
 * -# @ref CLWidthConcatenateLayerKernel (if underlying concatenation axis is 0).
 * -# @ref CLHeightConcatenateLayerKernel (if underlying concatenation axis is 1).
 * -# @ref CLDepthConcatenateLayerKernel (if underlying concatenation axis is 2).
 * -# @ref CLBatchConcatenateLayerKernel (if underlying concatenation axis is 3).
 */
class CLConcatenateLayer : public IFunction
{
public:
    /** Default constructor */
    CLConcatenateLayer();
    /** Destructor */
    ~CLConcatenateLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConcatenateLayer(const CLConcatenateLayer &) = delete;
    /** Default move constructor */
    CLConcatenateLayer(CLConcatenateLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConcatenateLayer &operator=(const CLConcatenateLayer &) = delete;
    /** Default move assignment operator */
    CLConcatenateLayer &operator=(CLConcatenateLayer &&);
    /** Initialise the kernel's inputs vector and output.
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref CLWidthConcatenateLayerKernel, @ref CLHeightConcatenateLayerKernel and @ref CLDepthConcatenateLayerKernel.
     *
     * @param[in,out] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: All
     * @param[out]    output        Output tensor. Data types supported: Same as @p input.
     * @param[in]     axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     */
    void configure(std::vector<const ICLTensor *> &inputs_vector, ICLTensor *output, size_t axis);
    /** Initialise the kernel's inputs vector and output.
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref CLWidthConcatenateLayerKernel, @ref CLHeightConcatenateLayerKernel and @ref CLDepthConcatenateLayerKernel.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] inputs_vector   The vectors containing all the tensors to concatenate. Data types supported: All
     * @param[out]    output          Output tensor. Data types supported: Same as @p input.
     * @param[in]     axis            Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     */
    void configure(const CLCompileContext &compile_context, std::vector<const ICLTensor *> &inputs_vector, ICLTensor *output, size_t axis);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConcatenateLayer
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref CLWidthConcatenateLayerKernel, @ref CLHeightConcatenateLayerKernel and @ref CLDepthConcatenateLayerKernel.
     *
     * @param[in] inputs_vector The vectors containing all the tensors info to concatenate. Data types supported: All.
     * @param[in] output        Output tensor info. Data types supported: Same as @p input.
     * @param[in] axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     *
     * @return a status
     */
    static Status validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

namespace experimental
{
/** Basic function to execute concatenate tensors along a given axis. This function calls the following kernels:
 *
 * -# @ref CLWidthConcatenateLayerKernel (if underlying concatenation axis is 0).
 * -# @ref CLHeightConcatenateLayerKernel (if underlying concatenation axis is 1).
 * -# @ref CLDepthConcatenateLayerKernel (if underlying concatenation axis is 2).
 * -# @ref CLBatchConcatenateLayerKernel (if underlying concatenation axis is 3).
 */
class CLConcatenation : public ICLOperator
{
public:
    /** Default constructor */
    CLConcatenation();
    /** Initialise the kernel's inputs vector and output.
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref CLWidthConcatenateLayerKernel, @ref CLHeightConcatenateLayerKernel and @ref CLDepthConcatenateLayerKernel.
     *
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in,out] inputs_vector   The vectors containing all the tensors to concatenate. Data types supported: All
     * @param[out]    output          Output tensor. Data types supported: Same as @p input.
     * @param[in]     axis            Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     */
    void configure(const CLCompileContext &compile_context, const std::vector<ITensorInfo *> &inputs_vector, ITensorInfo *output, size_t axis);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConcatenateLayer
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref CLWidthConcatenateLayerKernel, @ref CLHeightConcatenateLayerKernel and @ref CLDepthConcatenateLayerKernel.
     *
     * @param[in] inputs_vector The vectors containing all the tensors info to concatenate. Data types supported: All
     * @param[in] output        Output tensor info. Data types supported: Same as @p input.
     * @param[in] axis          Concatenation axis. Supported underlying concatenation axis are 0, 1, 2 and 3.
     *
     * @return a status
     */
    static Status validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output, size_t axis);

    // Inherited methods overridden:
    void run(InputTensorMap inputs, OutputTensorMap outputs, OperatorTensorMap workspace) override;

private:
    std::vector<std::unique_ptr<ICLKernel>> _concat_kernels;
    unsigned int                            _num_inputs;
    unsigned int                            _axis;
};
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLCONCATENATELAYER_H */
