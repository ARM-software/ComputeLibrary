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
#ifndef ARM_COMPUTE_CLSTACKLAYER_H
#define ARM_COMPUTE_CLSTACKLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>
#include <vector>

namespace arm_compute
{
class CLCompileContext;
class CLStackLayerKernel;
class ICLTensor;
class ITensorInfo;

/** Basic function to stack tensors along an axis. This function calls the following kernel:
 *
 * -# @ref CLStackLayerKernel
 *
 */
class CLStackLayer : public IFunction
{
public:
    /** Default constructor */
    CLStackLayer();
    /** Prevent instances of this class from being copied */
    CLStackLayer(const CLStackLayer &) = delete;
    /** Prevent instances of this class from being copied */
    CLStackLayer &operator=(const CLStackLayer &) = delete;
    /** Prevent instances of this class to be moved */
    CLStackLayer(CLStackLayer &&) = delete;
    /** Prevent instances of this class to be moved */
    CLStackLayer &operator=(CLStackLayer &&) = delete;
    /** Default destructor */
    ~CLStackLayer();
    /** Initialise the kernel's inputs vector and output.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in]  input  The vectors containing all the tensors with the same shape to stack. Data types supported: All.
     * @param[in]  axis   The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     *                    Negative values wrap around
     * @param[out] output Output tensor. Data types supported: Same as @p input.
     */
    void configure(const std::vector<ICLTensor *> &input, int axis, ICLTensor *output);
    /** Initialise the kernel's inputs vector and output.
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           The vectors containing all the tensors with the same shape to stack. Data types supported: All.
     * @param[in]  axis            The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     *                             Negative values wrap around
     * @param[out] output          Output tensor. Data types supported: Same as @p input.
     */
    void configure(const CLCompileContext &compile_context, const std::vector<ICLTensor *> &input, int axis, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLStackLayerKernel
     *
     * @note Supported input tensor rank: up to 4
     *
     * @param[in] input  The vectors containing all the tensors info with the same shape to stack. Data types supported: All.
     * @param[in] axis   The dimension to stack the tensors along. It must be smaller than the number of input dimensions.
     *                   Negative values wrap around
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const std::vector<ITensorInfo *> &input, int axis, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    std::vector<ICLTensor *>                         _input;
    std::vector<std::unique_ptr<CLStackLayerKernel>> _stack_kernels;
    unsigned int                                     _num_inputs;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLSTACKLAYER_H */
