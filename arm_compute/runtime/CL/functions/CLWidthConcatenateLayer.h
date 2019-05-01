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
#ifndef __ARM_COMPUTE_CLWIDTHCONCATENATELAYER_H__
#define __ARM_COMPUTE_CLWIDTHCONCATENATELAYER_H__

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLWidthConcatenate2TensorsKernel.h"
#include "arm_compute/core/CL/kernels/CLWidthConcatenate4TensorsKernel.h"
#include "arm_compute/core/CL/kernels/CLWidthConcatenateLayerKernel.h"

#include <memory>
#include <vector>

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute concatenate tensors along x axis. This function calls the following kernel:
 *
 * @deprecated This function is deprecated and will be removed in release 19.08
 *
 * -# @ref CLWidthConcatenateLayerKernel
 * -# @ref CLWidthConcatenate2TensorsKernel (if there are exactly 2 input tensors)
 * -# @ref CLWidthConcatenate4TensorsKernel (if there are exactly 4 input tensors)
 *
 */
class CLWidthConcatenateLayer : public IFunction
{
public:
    /** Default constructor */
    CLWidthConcatenateLayer();
    /** Initialise the kernel's inputs vector and output.
     *
     * @param[in]  inputs_vector The vectors containing all the tensors to concatenate. Data types supported: QASYMM8/F16/F32.
     *                           Dimensions of all the inputs should match apart for the width which can differ.
     * @param[out] output        Output tensor. Data types supported: Same as @p input.
     *                           Output tensor dimensions are the same with the inputs from the second dimension and above.
     *                           The first dimension (width) is the sum of the input tensors' widths.
     */
    void configure(std::vector<ICLTensor *> inputs_vector, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthConcatenateLayerKernel
     *
     * @param[in] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: QASYMM8/F16/F32.
     *                          Dimensions of all the inputs should match apart for the width which can differ.
     * @param[in] output        Output tensor. Data types supported: Same as @p input.
     *                          Output tensor dimensions are the same with the inputs from the second dimension and above.
     *                          The first dimension (width) is the sum of the input tensors' widths.
     *
     * @return a status
     */
    static Status validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    std::vector<CLWidthConcatenateLayerKernel> _concat_kernels_vector;
    CLWidthConcatenate2TensorsKernel           _concat_x2_kernel;
    CLWidthConcatenate4TensorsKernel           _concat_x4_kernel;
    unsigned int                               _num_inputs;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLWIDTHCONCATENATELAYER_H__ */
