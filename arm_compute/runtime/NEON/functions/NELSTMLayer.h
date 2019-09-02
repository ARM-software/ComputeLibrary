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
#ifndef __ARM_COMPUTE_NELSTMLAYER_H__
#define __ARM_COMPUTE_NELSTMLAYER_H__

#include "arm_compute/core/NEON/kernels/NEActivationLayerKernel.h"
#include "arm_compute/core/NEON/kernels/NEArithmeticAdditionKernel.h"
#include "arm_compute/core/NEON/kernels/NEArithmeticSubtractionKernel.h"
#include "arm_compute/core/NEON/kernels/NECopyKernel.h"
#include "arm_compute/core/NEON/kernels/NEPixelWiseMultiplicationKernel.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEMeanStdDevNormalizationLayer.h"
#include "arm_compute/runtime/common/LSTMParams.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to run @ref NELSTMLayer */
class NELSTMLayer : public IFunction
{
public:
    /** Default constructor */
    NELSTMLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Initialize function's tensors.
     *
     * @param[in]  input                       Source tensor. Input is a 2D tensor with dimensions [input_size, batch_size]. Data types supported: F16/F32.
     * @param[in]  input_to_forget_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  input_to_cell_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  input_to_output_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_forget_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_cell_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_output_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in]  forget_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in]  cell_bias                   1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in]  output_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in]  output_state_in             2D weights tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[in]  cell_state_in               2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[out] scratch_buffer              2D tensor with dimensions [num_units * 4, batch_size] with CIFG or [num_units * 3, batch_size] without CIGF. Data type supported: Same as @p input.
     * @param[out] output_state_out            2D weights tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[out] cell_state_out              2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[out] output                      Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size].
     *                                         Data types supported: Same as @p input.
     * @param[in]  lstm_params                 (Optional) Weights tensors used in peephole optimization:
     *                                         input_to_input_weights         2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     *                                         recurrent_to_input_weights     2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     *                                         cell_to_input_weights          1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: Same as @p input.
     *                                         cell_to_forget_weights         1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                         cell_to_output_weights         1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                         input_gate_bias                1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input
     *                                         projection_weights             2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     *                                         projection_bias                1D weights tensor with dimensions [output_size]. Data type supported: Same as @p input.
     *                                         input_layer_norm_coefficients  1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                         forget_layer_norm_coefficients 1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                         cell_layer_norm_coefficients   1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                         output_layer_norm_coefficients 1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in]  activation_info             Contains activation information described in @ref ActivationLayerInfo.
     * @param[in]  cell_threshold              The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip]. If set to 0.0 then clipping is disabled.
     * @param[in]  projection_threshold        The clipping threshold for the output from the projection layer, such that values are bound within [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     */
    void configure(const ITensor *input,
                   const ITensor *input_to_forget_weights, const ITensor *input_to_cell_weights, const ITensor *input_to_output_weights,
                   const ITensor *recurrent_to_forget_weights, const ITensor *recurrent_to_cell_weights, const ITensor *recurrent_to_output_weights,
                   const ITensor *forget_gate_bias, const ITensor *cell_bias, const ITensor *output_gate_bias,
                   const ITensor *output_state_in, const ITensor *cell_state_in,
                   ITensor *scratch_buffer, ITensor *output_state_out, ITensor *cell_state_out, ITensor *output,
                   const LSTMParams<ITensor> &lstm_params, const ActivationLayerInfo &activation_info, float cell_threshold = 0.f, float projection_threshold = 0.f);

    /** Static function to check if given info will lead to a valid configuration of @ref NELSTMLayer
     *
     * @param[in] input                       Source tensor. Input is a 2D tensor with dimensions [input_size, batch_size]. Data types supported: F16/F32.
     * @param[in] input_to_forget_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in] input_to_cell_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in] input_to_output_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     * @param[in] recurrent_to_forget_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in] recurrent_to_cell_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in] recurrent_to_output_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     * @param[in] forget_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in] cell_bias                   1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in] output_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in] output_state_in             2D weights tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[in] cell_state_in               2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[in] scratch_buffer              2D tensor with dimensions [num_units * 4, batch_size] with CIFG or [num_units * 3, batch_size] without CIGF. Data type supported: Same as @p input.
     * @param[in] output_state_out            2D weights tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[in] cell_state_out              2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[in] output                      Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size].
     *                                        Data types supported: Same as @p input.
     * @param[in] lstm_params                 (Optional) Weights tensors used in peephole optimization:
     *                                        input_to_input_weights         2D weights tensor with dimensions [input_size, num_units]. Data type supported: Same as @p input.
     *                                        recurrent_to_input_weights     2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     *                                        cell_to_input_weights          1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: Same as @p input.
     *                                        cell_to_forget_weights         1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                        cell_to_output_weights         1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input.
     *                                        input_gate_bias                1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input
     *                                        projection_weights             2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input.
     *                                        projection_bias                1D weights tensor with dimensions [output_size]. Data type supported: Same as @p input.
     *                                        input_layer_norm_coefficients  1D weights tensor info with dimensions [num_units]. Data type supported: Same as @p input.
     *                                        forget_layer_norm_coefficients 1D weights tensor info with dimensions [num_units]. Data type supported: Same as @p input.
     *                                        cell_layer_norm_coefficients   1D weights tensor info with dimensions [num_units]. Data type supported: Same as @p input.
     *                                        output_layer_norm_coefficients 1D weights tensor info with dimensions [num_units]. Data type supported: Same as @p input.
     * @param[in] activation_info             Contains activation information described in @ref ActivationLayerInfo.
     * @param[in] cell_threshold              The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip]. If set to 0.0 then clipping is disabled.
     * @param[in] projection_threshold        The clipping threshold for the output from the projection layer, such that values are bound within [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input,
                           const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                           const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                           const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                           const ITensorInfo *output_state_in, const ITensorInfo *cell_state_in,
                           const ITensorInfo *scratch_buffer, const ITensorInfo *output_state_out, const ITensorInfo *cell_state_out, const ITensorInfo *output,
                           const LSTMParams<ITensorInfo> &lstm_params, const ActivationLayerInfo &activation_info, float cell_threshold = 0.f, float projection_threshold = 0.f);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup                     _memory_group;
    NEFullyConnectedLayer           _fully_connected_input_gate;
    NEArithmeticAddition            _accum_input_gate1;
    NEArithmeticSubtractionKernel   _subtract_input_gate;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_input_gate;
    NEActivationLayerKernel         _activation_input_gate;
    NEFullyConnectedLayer           _fully_connected_forget_gate;
    NEArithmeticAddition            _accum_forget_gate1;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_forget_gate;
    NEActivationLayerKernel         _activation_forget_gate;
    NEFullyConnectedLayer           _fully_connected_cell_state;
    NEGEMM                          _gemm_cell_state1;
    NETransposeKernel               _transpose_cell_state;
    NEArithmeticAdditionKernel      _accum_cell_state1;
    NEArithmeticAdditionKernel      _accum_cell_state2;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_cell_state1;
    NEActivationLayerKernel         _activation_cell_state;
    NEActivationLayerKernel         _cell_clip;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_cell_state2;
    NEFullyConnectedLayer           _fully_connected_output;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_output_state1;
    NEArithmeticAddition            _accum_output1;
    NEActivationLayerKernel         _activation_output;
    NEActivationLayerKernel         _activation_output_state;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_output_state2;
    NEFullyConnectedLayer           _fully_connected_output_state;
    NEActivationLayerKernel         _projection_clip;
    NECopyKernel                    _copy_cell_state;
    NECopyKernel                    _copy_output;
    NEConcatenateLayer              _concat_scratch_buffer;
    NEConcatenateLayer              _concat_inputs_forget_gate;
    NEConcatenateLayer              _concat_weights_forget_gate;
    NEConcatenateLayer              _concat_weights_input_gate;
    NEConcatenateLayer              _concat_weights_output;
    NEMeanStdDevNormalizationLayer  _mean_std_norm_input_gate;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_input_gate_coeff;
    NEArithmeticAdditionKernel      _accum_input_gate_bias;
    NEMeanStdDevNormalizationLayer  _mean_std_norm_forget_gate;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_forget_gate_coeff;
    NEArithmeticAdditionKernel      _accum_forget_gate_bias;
    NEMeanStdDevNormalizationLayer  _mean_std_norm_cell_gate;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_cell_gate_coeff;
    NEArithmeticAdditionKernel      _accum_cell_gate_bias;
    NEMeanStdDevNormalizationLayer  _mean_std_norm_output_gate;
    NEPixelWiseMultiplicationKernel _pixelwise_mul_output_gate_coeff;
    NEArithmeticAdditionKernel      _accum_output_gate_bias;
    Tensor                          _input_gate_out1;
    Tensor                          _input_gate_out2;
    Tensor                          _input_gate_out3;
    Tensor                          _input_gate_out4;
    Tensor                          _forget_gate_out1;
    Tensor                          _forget_gate_out2;
    Tensor                          _forget_gate_out3;
    Tensor                          _forget_gate_out4;
    Tensor                          _forget_gate_out5;
    Tensor                          _forget_gate_out6;
    Tensor                          _cell_state_out1;
    Tensor                          _cell_state_out2;
    Tensor                          _cell_state_out3;
    Tensor                          _cell_state_out4;
    Tensor                          _cell_state_out5;
    Tensor                          _output1;
    Tensor                          _output2;
    Tensor                          _output3;
    Tensor                          _output4;
    Tensor                          _cell_state_activation;
    Tensor                          _output_state1;
    Tensor                          _ones;
    Tensor                          _input_layer_norm_out1;
    Tensor                          _input_layer_norm_out2;
    Tensor                          _forget_layer_norm_out1;
    Tensor                          _forget_layer_norm_out2;
    Tensor                          _cell_layer_norm_out1;
    Tensor                          _cell_layer_norm_out2;
    Tensor                          _output_layer_norm_out1;
    Tensor                          _output_layer_norm_out2;
    bool                            _run_peephole_opt;
    bool                            _run_cifg_opt;
    bool                            _perform_cell_clipping;
    bool                            _has_projection_weights;
    bool                            _perform_projection_clipping;
    bool                            _is_prepared;
    bool                            _is_layer_norm_lstm;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NELSTMLAYER_H__ */
