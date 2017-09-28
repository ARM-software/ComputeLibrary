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
#ifndef __ARM_COMPUTE_NELOCALLYCONNECTEDLAYER_H__
#define __ARM_COMPUTE_NELOCALLYCONNECTEDLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NECol2ImKernel.h"
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"
#include "arm_compute/core/NEON/kernels/NELocallyConnectedMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEWeightsReshapeKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class INETensor;

/** Basic function to compute the locally connected layer. This function calls the following NEON kernels:
 *
 * -# @ref NEWeightsReshapeKernel (executed only once for each configuration)
 * -# @ref NEIm2ColKernel
 * -# @ref NELocallyConnectedMatrixMultiplyKernel
 * -# @ref NECol2ImKernel
 */
class NELocallyConnectedLayer : public IFunction
{
public:
    /** Default constructor */
    NELocallyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs.
     *                       Data types supported: F16, F32.
     * @param[in]  weights   Weights tensor. Weights are 5D tensor with dimensions [kernel_x, kernel_y, IFM, OFM, num_patches]. Data type supported:Same as @p input.
     * @param[in]  biases    Biases tensor. Shared biases supported. Biases are 2D tensor with dimensions [OFM, num_patches]. Data type supported:Same as @p input.
     * @param[out] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                       Data types supported: Same as @p input.
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                            _memory_group;
    NEIm2ColKernel                         _input_im2col_kernel;
    NEWeightsReshapeKernel                 _weights_reshape_kernel;
    NELocallyConnectedMatrixMultiplyKernel _mm_kernel;
    NECol2ImKernel                         _output_col2im_kernel;
    Tensor                                 _input_im2col_reshaped;
    Tensor                                 _weights_reshaped;
    Tensor                                 _gemm_output;
    bool                                   _is_first_run;
};
}
#endif /* __ARM_COMPUTE_NELOCALLYCONNECTEDLAYER_H__ */
