/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEDEPTHCONCATENATE_H__
#define __ARM_COMPUTE_NEDEPTHCONCATENATE_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEDepthConcatenateLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"

#include <memory>
#include <vector>

namespace arm_compute
{
class ITensor;

/** Basic function to execute concatenate tensors along z axis. This function calls the following kernels:
 *
 * -# @ref NEFillBorderKernel (executed if input's lowest two dimensions are smaller than respective output's dimensions)
 * -# @ref NEDepthConcatenateLayerKernel
 *
 */
class NEDepthConcatenateLayer : public IFunction
{
public:
    /** Default constructor */
    NEDepthConcatenateLayer();
    /** Initialise the kernel's inputs vector and output.
     *
     * @param[in,out] inputs_vector The vectors containing all the tensors to concatenate. Data types supported:  QS8/QS16/F16/F32.
     * @param[out]    output        Output tensor. Data types supported: Same as @p inputs_vector.
     */
    void configure(std::vector<ITensor *> inputs_vector, ITensor *output);

    // Inherited methods overridden:
    void run() override;

private:
    std::vector<ITensor *>                           _inputs_vector;
    std::unique_ptr<NEDepthConcatenateLayerKernel[]> _concat_kernels_vector;
    std::unique_ptr<NEFillBorderKernel[]>            _border_handlers_vector;
    unsigned int                                     _num_inputs;
};
}
#endif /* __ARM_COMPUTE_NEDEPTHCONCATENATE_H__ */
