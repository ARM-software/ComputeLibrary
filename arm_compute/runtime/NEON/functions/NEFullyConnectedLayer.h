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
#ifndef __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__
#define __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"
#include "arm_compute/core/NEON/kernels/NETransposeKernel.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
/** Basic function to compute a Fully Connected layer on NEON. This function calls the following NEON kernels:
 *  -# @ref NEIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref NETransposeKernel (if @p transpose_weights flag is set to true) (called once)
 *  -# @ref NEGEMMTranspose1xWKernel (called once if we have a multi-batch input)
 *  -# @ref NEGEMMInterleave4x4Kernel (called if we have a multi-batch input)
 *  -# @ref NEGEMMMatrixMultiplyKernel
 *  -# @ref NEGEMMMatrixAccumulateBiasesKernel (if @p biases is not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class NEFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    NEFullyConnectedLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input             Source tensor. Data type supported: F32.
     * @param[in]  weights           Weights tensor. The weights must be 2 dimensional. Data type supported: Same as @p input.
     * @param[in]  biases            Bias tensor. Can be nullptr. Data type supported:Same as @p input.
     * @param[out] output            Destination tensor. Data type supported: Same as @p input.
     * @param[in]  transpose_weights (Optional) Transpose weights if true. Defaults to true.
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, bool transpose_weights = true);

    //Inherited methods override
    void run() override;

private:
    void configure_fc_fc_wb(const ITensor *input, const ITensor *weights, ITensor *output);
    void configure_fc_fc_nb(const ITensor *input, const ITensor *weights, ITensor *output);
    void configure_conv_fc_wb(const ITensor *input, const ITensor *weights, ITensor *output);
    void configure_conv_fc_nb(const ITensor *input, const ITensor *weights, ITensor *output);

    NEIm2ColKernel                     _im2col_kernel;
    NETransposeKernel                  _transpose_kernel;
    NEGEMMTranspose1xWKernel           _transpose1xW_kernel;
    NEGEMMInterleave4x4Kernel          _interleave4x4_kernel;
    NEGEMMMatrixMultiplyKernel         _mm_kernel;
    NEGEMMMatrixAccumulateBiasesKernel _accumulate_biases_kernel;
    Tensor                             _im2col_output;
    Tensor                             _interleave4x4_output;
    Tensor                             _transpose_output;
    Tensor                             _transpose1xW_output;
    bool                               _is_first_run;
    bool                               _transpose_weights;
    bool                               _fc_after_conv;
    bool                               _batched_fc_layer;
    bool                               _accumulate_biases;
};
}
#endif /* __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__ */
