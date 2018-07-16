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
#ifndef __ARM_COMPUTE_NECONCATENATELAYER_H__
#define __ARM_COMPUTE_NECONCATENATELAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

#include <memory>
#include <vector>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;
class Status;

/** Basic function to execute concatenate tensors along a given axis. This function calls the following kernels:
 *
 * -# @ref NEWidthConcatenateLayer (if underlying concatenation axis is 0).
 * -# @ref NEDepthConcatenateLayer (if underlying concatenation axis is 2).
 */
class NEConcatenateLayer : public IFunction
{
public:
    /** Default constructor */
    NEConcatenateLayer();
    /** Initialise the kernel's inputs vector and output.
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref NEWidthConcatenateLayer and @ref NEDepthConcatenateLayer.
     *
     * @param[in,out] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: QASYMM8/F16/F32.
     * @param[out]    output        Output tensor. Data types supported: Same as @p input.
     * @param[in]     axis          Concatenation axis. Supported underlying concatenation axis are 0 and 2.
     */
    void configure(const std::vector<ITensor *> &inputs_vector, ITensor *output, DataLayoutDimension axis);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConcatenateLayer
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     * @note Preconditions can be found respectively at @ref NEWidthConcatenateLayer and @ref NEDepthConcatenateLayer.
     *
     * @param[in] inputs_vector The vectors containing all the tensors info to concatenate. Data types supported: QASYMM8/F16/F32.
     * @param[in] output        Output tensor info. Data types supported: Same as @p input.
     * @param[in] axis          Concatenation axis. Supported underlying concatenation axis are 0 and 2.
     *
     * @return a status
     */
    static Status validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output, DataLayoutDimension axis);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<IFunction> _concat_function;
};
}
#endif /* __ARM_COMPUTE_NECONCATENATELAYER_H__ */
