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
#ifndef __ARM_COMPUTE_NERNNLAYER_H__
#define __ARM_COMPUTE_NERNNLAYER_H__

#include "arm_compute/core/NEON/kernels/NEActivationLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEArithmeticAdditionKernel.h"
#include "arm_compute/core/NEON/kernels/NECopyKernel.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to run @ref NERNNLayer */
class NERNNLayer : public IFunction
{
public:
    /** Default constructor */
    NERNNLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NERNNLayer(const NERNNLayer &) = delete;
    /** Default move constructor */
    NERNNLayer(NERNNLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NERNNLayer &operator=(const NERNNLayer &) = delete;
    /** Default move assignment operator */
    NERNNLayer &operator=(NERNNLayer &&) = default;
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
    void configure(const ITensor *input, const ITensor *weights, const ITensor *recurrent_weights, const ITensor *bias, ITensor *hidden_state, ITensor *output, ActivationLayerInfo &info);
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
    MemoryGroup                _memory_group;
    NEGEMM                     _gemm_state_f;
    NEArithmeticAdditionKernel _add_kernel;
    NEActivationLayerKernel    _activation_kernel;
    NEFullyConnectedLayer      _fully_connected_kernel;
    NECopyKernel               _copy_kernel;
    Tensor                     _fully_connected_out;
    Tensor                     _gemm_output;
    Tensor                     _add_output;
    bool                       _is_prepared;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NERNNLAYER_H__ */
