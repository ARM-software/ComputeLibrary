/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GCDEPTHCONCATENATELAYER_H
#define ARM_COMPUTE_GCDEPTHCONCATENATELAYER_H

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDepthConcatenateLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>
#include <vector>

namespace arm_compute
{
class IGCTensor;

/** Basic function to execute concatenate tensors along a given axis. This function calls the following kernels:
 *
 * @note only axis z is supported
 * -# @ref GCDepthConcatenateLayerKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCConcatenateLayer : public IFunction
{
public:
    /** Default constructor */
    GCConcatenateLayer();
    /** Initialise the kernel's inputs vector and output.
     *
     * @note Input and output tensor dimensions preconditions defer depending on the concatenation axis.
     *
     * @param[in,out] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: F16/F32.
     * @param[out]    output        Output tensor. Data types supported: Same as @p input.
     * @param[in]     axis          Concatenation axis. Supported underlying concatenation axis is 2.
     */
    void configure(std::vector<IGCTensor *> inputs_vector, IGCTensor *output, size_t axis);

    // Inherited methods overridden:
    void run() override;

private:
    std::vector<std::unique_ptr<IGCKernel>> _concat_kernels;
    unsigned int                            _num_inputs;
    unsigned int                            _axis;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_GCDEPTHCONCATENATELAYER_H */
