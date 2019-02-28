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
#ifndef __ARM_COMPUTE_CLRNN_LAYER_H__
#define __ARM_COMPUTE_CLRNN_LAYER_H__

#include "arm_compute/core/CL/kernels/CLActivationLayerKernel.h"
#include "arm_compute/core/CL/kernels/CLCopyKernel.h"
#include "arm_compute/core/CL/kernels/CLElementwiseOperationKernel.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLRNNLayer */
class CLRNNLayer : public IFunction
{
public:
    /** Default constructor */
    CLRNNLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialize the function
     *
     * @param[in]     input             Input is a 2-D tensor of shape [input_size, batch_size]. Data types supported: F16/F32
     * @param[in]     weights           Weights tensor of shape [input_size, num_units] that multiplies the input. Data types supported: Same as @p input
     * @param[in]     recurrent_weights Weights tensor of shape [num_units, num_units] that multiplies the current 'state'. Data types supported: Same as @p input
     * @param[in]     bias              Bias vector of shape [num_units]. Data types supported: Same as @p input
     * @param[out]    output            Output tensor of shape [num_units, batch_size]. Data types supported: Same as @p input
     * @param[in,out] hidden_state      Output tensor of shape [num_units, batch_size]. Data types supported: Same as @p input
     * @param[in]     info              Activation layer parameter.
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *recurrent_weights, const ICLTensor *bias, ICLTensor *hidden_state, ICLTensor *output, ActivationLayerInfo &info);
    /** Initialize the function
     *
     * @param[in] input             Input is a 2-D tensor of shape [input_size, batch_size]. Data types supported: F16/F32
     * @param[in] weights           Weights tensor of shape [input_size, num_units] that multiplies the input. Data types supported: Same as @p input
     * @param[in] recurrent_weights Weights tensor of shape [num_units, num_units] that multiplies the current 'state'. Data types supported: Same as @p input
     * @param[in] bias              Bias vector of shape [num_units]. Data types supported: Same as @p input
     * @param[in] output            Output tensor of shape [num_units, batch_size]. Data types supported: Same as @p input
     * @param[in] hidden_state      Output tensor of shape [num_units, batch_size]. Data types supported: Same as @p input
     * @param[in] info              Activation layer parameter.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *recurrent_weights, const ITensorInfo *bias, const ITensorInfo *hidden_state, const ITensorInfo *output,
                           const ActivationLayerInfo &info);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    CLMemoryGroup                        _memory_group;
    CLGEMM                               _gemm_state_f;
    CLSaturatedArithmeticOperationKernel _add_kernel;
    CLActivationLayerKernel              _activation_kernel;
    CLFullyConnectedLayer                _fully_connected_kernel;
    CLCopyKernel                         _copy_kernel;
    CLTensor                             _fully_connected_out;
    CLTensor                             _gemm_output;
    CLTensor                             _add_output;
    bool                                 _is_prepared;
};
}
#endif /* __ARM_COMPUTE_CLRNN_LAYER_H__ */
