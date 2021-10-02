/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPP_SPLIT_H
#define ARM_COMPUTE_CPP_SPLIT_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
/** Basic function to split a tensor along a given axis */
template <typename SliceType, typename TensorInterfaceType = ITensor>
class CPPSplit : public IFunction
{
public:
    CPPSplit();

    /** Static function to check if given info will lead to a valid configuration of @ref CPPSplit
     *
     * @param[in] input   The input tensor info. Data types supported: All.
     * @param[in] outputs A vector containing the output tensors' info. Data types supported: same as @p input.
     *                    The output tensors should match the input tensor dimensions for all shape dimensions apart
     *                    from the split dimension
     * @param[in] axis    Axis on which to split the input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const std::vector<ITensorInfo *> &outputs, unsigned int axis);

    /** Initialise the kernel's input and outputs.
     *
     * @param[in]  input   The input tensor. Data types supported: All
     * @param[out] outputs A vector containing the output tensors. Data types supported: Same as @p input.
     *                     The output tensors should match the input tensor dimensions for all shape dimensions apart
     *                     from the split dimension.
     * @param[in]  axis    Axis on which to split the input.
     */
    void configure(const TensorInterfaceType *input, const std::vector<TensorInterfaceType *> &outputs, unsigned int axis);

protected:
    std::vector<TensorInterfaceType *> _outputs_vector;
    std::vector<SliceType>             _slice_functions;
    unsigned int                       _num_outputs;
};

} // namespace arm_compute
#endif /* ARM_COMPUTE_CPP_SPLIT_H */
