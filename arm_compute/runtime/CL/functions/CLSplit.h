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
#ifndef __ARM_COMPUTE_CLSPLIT_H__
#define __ARM_COMPUTE_CLSPLIT_H__

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/CL/functions/CLSlice.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>
#include <vector>

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Basic function to split a tensor along a given axis */
class CLSplit : public IFunction
{
public:
    /** Default constructor */
    CLSplit();
    /** Initialise the kernel's input and outputs.
     *
     * @param[in]  input   The input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[out] outputs A vector containing the output tensors. Data types supported: Same as @p input.
     *                     The output tensors should match the input tensor dimensions for all shape dimensions apart
     *                     from the split dimension.
     * @param[in]  axis    Axis on which to split the input.
     */
    void configure(const ICLTensor *input, const std::vector<ICLTensor *> &outputs, unsigned int axis);
    /** Static function to check if given info will lead to a valid configuration of @ref CLSplit
     *
     * @param[in] input   The input tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in] outputs A vector containing the output tensors' info. Data types supported: Same as @p input.
     *                    The output tensors should match the input tensor dimensions for all shape dimensions apart
     *                    from the split dimension
     * @param[in] axis    Axis on which to split the input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const std::vector<ITensorInfo *> &outputs, unsigned int axis);

    // Inherited methods overridden:
    void run() override;

private:
    std::vector<ICLTensor *> _outputs_vector;
    std::vector<CLSlice>     _slice_functions;
    unsigned int             _num_outputs;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLSPLIT_H__ */
