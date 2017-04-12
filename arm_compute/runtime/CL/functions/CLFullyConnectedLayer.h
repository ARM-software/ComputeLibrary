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
#ifndef __ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H__
#define __ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"
#include "arm_compute/core/CL/kernels/CLIm2ColKernel.h"
#include "arm_compute/core/CL/kernels/CLTransposeKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"

namespace arm_compute
{
/** Basic function to compute a Fully Connected layer on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref CLIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref CLTransposeKernel (if @p transpose_weights is set to true) (called once)
 *  -# @ref NEGEMMTranspose1xWKernel (called once if we have a multi-batch input)
 *  -# @ref NEGEMMInterleave4x4Kernel (called if we have a multi-batch input)
 *  -# @ref NEGEMMMatrixMultiplyKernel
 *  -# @ref CLGEMMMatrixAccumulateBiasesKernel (if @p biases is not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class CLFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    CLFullyConnectedLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input             Source tensor. Data type supported: F16, F32.
     * @param[in]  weights           Weights tensor. The weights must be 2 dimensional. Data type supported: Same as @p input
     * @param[in]  biases            Bias tensor. It can be nullptr. Data type supported:Same as @p input.
     * @param[out] output            Destination tensor. Data type supported: Same as @p input.
     * @param[in]  transpose_weights (Optional) Transpose weights if true. Defaults to true.
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, bool transpose_weights = true);

    //Inherited methods override
    void run() override;

private:
    void configure_fc_fc_wb(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output);
    void configure_fc_fc_nb(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output);
    void configure_conv_fc_wb(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output);
    void configure_conv_fc_nb(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output);

    CLIm2ColKernel                     _im2col_kernel;
    CLTransposeKernel                  _transpose_kernel;
    CLGEMMTranspose1xWKernel           _transpose1xW_kernel;
    CLGEMMInterleave4x4Kernel          _interleave4x4_kernel;
    CLGEMMMatrixMultiplyKernel         _mm_kernel;
    CLGEMMMatrixAccumulateBiasesKernel _accumulate_biases_kernel;
    CLTensor                           _im2col_output;
    CLTensor                           _interleave4x4_output;
    CLTensor                           _transpose_output;
    CLTensor                           _transpose1xW_output;
    bool                               _is_first_run;
    bool                               _transpose_weights;
    bool                               _fc_after_conv;
    bool                               _batched_fc_layer;
    bool                               _accumulate_biases;
};
}
#endif /* __ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H__ */
