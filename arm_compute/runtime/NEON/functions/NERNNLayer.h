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
#ifndef ARM_COMPUTE_NERNNLAYER_H
#define ARM_COMPUTE_NERNNLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/NEON/functions/NECopy.h"
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
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NERNNLayer(NERNNLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NERNNLayer &operator=(const NERNNLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NERNNLayer &operator=(NERNNLayer &&) = delete;
    /** Default destructor */
    ~NERNNLayer();
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
    MemoryGroup           _memory_group;
    NEGEMM                _gemm_state_f;
    NEArithmeticAddition  _add_f;
    NEActivationLayer     _activation;
    NEFullyConnectedLayer _fully_connected;
    NECopy                _copy_f;
    Tensor                _fully_connected_out;
    Tensor                _gemm_output;
    Tensor                _add_output;
    bool                  _is_prepared;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NERNNLAYER_H */
