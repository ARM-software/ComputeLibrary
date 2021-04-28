/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLRNN_LAYER_H
#define ARM_COMPUTE_CLRNN_LAYER_H

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLCopy.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLRNNLayer */
class CLRNNLayer : public IFunction
{
public:
    /** Default constructor */
    CLRNNLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied */
    CLRNNLayer(const CLRNNLayer &) = delete;
    /** Prevent instances of this class from being copied */
    CLRNNLayer &operator=(const CLRNNLayer &) = delete;
    /** Default destructor */
    ~CLRNNLayer();
    /** Initialize the function
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0   |src1   |src2   |src3   |dst0   |dst1   |
     * |:------|:------|:------|:------|:------|:------|
     * |F16    |F16    |F16    |F16    |F16    |F16    |
     * |F32    |F32    |F32    |F32    |F32    |F32    |
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
     * @param[in]     compile_context   The compile context to be used.
     * @param[in]     input             Input is a 2-D tensor of shape [input_size, batch_size]. Data types supported: F16/F32
     * @param[in]     weights           Weights tensor of shape [input_size, num_units] that multiplies the input. Data types supported: Same as @p input
     * @param[in]     recurrent_weights Weights tensor of shape [num_units, num_units] that multiplies the current 'state'. Data types supported: Same as @p input
     * @param[in]     bias              Bias vector of shape [num_units]. Data types supported: Same as @p input
     * @param[out]    output            Output tensor of shape [num_units, batch_size]. Data types supported: Same as @p input
     * @param[in,out] hidden_state      Output tensor of shape [num_units, batch_size]. Data types supported: Same as @p input
     * @param[in]     info              Activation layer parameter.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *recurrent_weights, const ICLTensor *bias, ICLTensor *hidden_state,
                   ICLTensor *output, ActivationLayerInfo &info);
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
    MemoryGroup           _memory_group;
    CLGEMM                _gemm_state_f;
    CLArithmeticAddition  _add_kernel;
    CLActivationLayer     _activation;
    CLFullyConnectedLayer _fully_connected_kernel;
    CLCopy                _copy;
    CLTensor              _fully_connected_out;
    CLTensor              _gemm_output;
    CLTensor              _add_output;
    bool                  _is_prepared;
};
}
#endif /* ARM_COMPUTE_CLRNN_LAYER_H */
