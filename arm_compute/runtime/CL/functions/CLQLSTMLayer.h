/*
 * Copyright (c) 2020 ARM Limited.
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
#ifndef ARM_COMPUTE_CLQLSTMLAYER_H
#define ARM_COMPUTE_CLQLSTMLAYER_H

#include "arm_compute/core/CL/kernels/CLElementwiseOperationKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMLowpReductionKernel.h"
#include "arm_compute/core/CL/kernels/CLPixelWiseMultiplicationKernel.h"
#include "arm_compute/core/CL/kernels/CLQLSTMLayerNormalizationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "arm_compute/runtime/CL/functions/CLTranspose.h"

#include "arm_compute/runtime/common/LSTMParams.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Basic function to run @ref CLQLSTMLayer
 *
 * This function calls the following CL functions/kernels:
 *
 * -# @ref CLActivationLayer                                     Activation functions (tanh and logistic)
 * -# @ref CLSaturatedArithmeticOperationKernel                  Elementwise addition and subtraction
 * -# @ref CLGEMMLowpMatrixMultiplyCore                          Quantized matrix multiplication core. Accumulators are 32-bit integers
 * -# @ref CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint   Convert 32-bit integers into QSYMM16
 * -# @ref CLGEMMLowpMatrixAReductionKernel                      For precomputing effective biases to use
 * -# @ref CLPixelWiseMultiplicationKernel                       Elementwise multiplication
 * -# @ref CLTranspose                                           Transpose function for reshaping the weights
 * */
class CLQLSTMLayer : public IFunction
{
public:
    /** Default constructor */
    CLQLSTMLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLQLSTMLayer(const CLQLSTMLayer &) = delete;
    /** Default move constructor */
    CLQLSTMLayer(CLQLSTMLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLQLSTMLayer &operator=(const CLQLSTMLayer &) = delete;
    /** Default move assignment operator */
    CLQLSTMLayer &operator=(CLQLSTMLayer &&) = default;
    /** Initialize function's tensors.
     *
     * @param[in]  input                       Source tensor. Input is a 2D tensor with dimensions [input_size, batch_size]. Data types supported: QASYMM8_SIGNED.
     * @param[in]  input_to_forget_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  input_to_cell_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  input_to_output_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_forget_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_cell_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_output_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  forget_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  cell_bias                   1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  output_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  cell_state_in               2D tensor with dimensions [output_size, batch_size]. Data type supported:  QSYMM16.
     * @param[in]  output_state_in             2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[out] cell_state_out              Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size]. Data type supported:  QSYMM16.
     * @param[out] output_state_out            Destination tensor. Output is a 2D tensor with dimensions [num_units, batch_size].Data types supported: Same as @p input.
     * @param[in]  lstm_params                 Weights tensors used in peephole, CIFG and layer normalization optimizations:
     *                                         input_intermediate_scale   Scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
     *                                         forget_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
     *                                         cell_intermediate_scale    Scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
     *                                         output_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
     *                                         hidden_state_zero          The zero point of the hidden state.
     *                                         hidden_state_scale         The scale of the hidden state.
     *                                         input_to_input_weights     (Optional) 2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     *                                         recurrent_to_input_weights (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                         cell_to_input_weights      (Optional) 1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: QSYMM16.
     *                                         cell_to_forget_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_to_output_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         input_gate_bias            (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: S32.
     *                                         projection_weights         (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                         projection_bias            (Optional) 1D weights tensor with dimensions [output_size]. S32.
     *                                         input_layer_norm_weights   (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         forget_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_layer_norm_weights    (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         output_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_threshold             (Optional) The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip].
     *                                                                               If set to 0.0 then clipping is disabled.
     *                                         projection_threshold       (Optional) The clipping threshold for the output from the projection layer, such that values are bound within
     *                                                                               [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     */
    void configure(const ICLTensor *input,
                   const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                   const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                   const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                   const ICLTensor *cell_state_in, const ICLTensor *output_state_in,
                   ICLTensor *cell_state_out, ICLTensor *output_state_out,
                   const LSTMParams<ICLTensor> &lstm_params);

    /** Initialize function's tensors.
     *
     * @param[in]  compile_context             The compile context to be used.
     * @param[in]  input                       Source tensor. Input is a 2D tensor with dimensions [input_size, batch_size]. Data types supported: QASYMM8_SIGNED.
     * @param[in]  input_to_forget_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  input_to_cell_weights       2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  input_to_output_weights     2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_forget_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_cell_weights   2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_output_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  forget_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  cell_bias                   1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  output_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: S32.
     * @param[in]  cell_state_in               2D tensor with dimensions [output_size, batch_size]. Data type supported:  QSYMM16.
     * @param[in]  output_state_in             2D tensor with dimensions [num_units, batch_size]. Data type supported: Same as @p input.
     * @param[out] cell_state_out              Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size]. Data type supported:  QSYMM16.
     * @param[out] output_state_out            Destination tensor. Output is a 2D tensor with dimensions [num_units, batch_size].Data types supported: Same as @p input.
     * @param[in]  lstm_params                 Weights tensors used in peephole, CIFG and layer normalization optimizations:
     *                                         input_intermediate_scale   Scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
     *                                         forget_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
     *                                         cell_intermediate_scale    Scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
     *                                         output_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
     *                                         hidden_state_zero          The zero point of the hidden state.
     *                                         hidden_state_scale         The scale of the hidden state.
     *                                         input_to_input_weights     (Optional) 2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     *                                         recurrent_to_input_weights (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                         cell_to_input_weights      (Optional) 1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: QSYMM16.
     *                                         cell_to_forget_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_to_output_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         input_gate_bias            (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: S32.
     *                                         projection_weights         (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                         projection_bias            (Optional) 1D weights tensor with dimensions [output_size]. S32.
     *                                         input_layer_norm_weights   (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         forget_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_layer_norm_weights    (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         output_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_threshold             (Optional) The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip].
     *                                                                               If set to 0.0 then clipping is disabled.
     *                                         projection_threshold       (Optional) The clipping threshold for the output from the projection layer, such that values are bound within
     *                                                                               [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input,
                   const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                   const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                   const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                   const ICLTensor *cell_state_in, const ICLTensor *output_state_in,
                   ICLTensor *cell_state_out, ICLTensor *output_state_out,
                   const LSTMParams<ICLTensor> &lstm_params);

    /** Static function to check if given info will lead to a valid configuration of @ref CLQLSTMLayer
     *
     * @param[in]  input                       Source tensor info. Input is a 2D tensor info with dimensions [input_size, batch_size]. Data types supported: QASYMM8_SIGNED.
     * @param[in]  input_to_forget_weights     2D weights tensor info with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  input_to_cell_weights       2D weights tensor info with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  input_to_output_weights     2D weights tensor info with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_forget_weights 2D weights tensor info with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_cell_weights   2D weights tensor info with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  recurrent_to_output_weights 2D weights tensor info with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     * @param[in]  forget_gate_bias            1D weights tensor info with dimensions [num_units]. Data type supported: S32.
     * @param[in]  cell_bias                   1D weights tensor info with dimensions [num_units]. Data type supported: S32.
     * @param[in]  output_gate_bias            1D weights tensor info with dimensions [num_units]. Data type supported: S32.
     * @param[in]  cell_state_in               2D tensor info with dimensions [num_units, batch_size]. Data type supported:  QSYMM16.
     * @param[in]  output_state_in             2D tensor info with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[out] cell_state_out              Destination tensor info. Output is a 2D tensor info with dimensions [num_units, batch_size]. Data type supported:  QSYMM16.
     * @param[out] output_state_out            Destination tensor info. Output is a 2D tensor info with dimensions [output_size, batch_size].Data types supported: Same as @p input.
     * @param[in]  lstm_params                 Weights tensors info used in peephole, CIFG and layer normalization optimizations:
     *                                         input_intermediate_scale   Scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
     *                                         forget_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
     *                                         cell_intermediate_scale    Scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
     *                                         output_intermediate_scale  Scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
     *                                         hidden_state_zero          The zero point of the hidden state.
     *                                         hidden_state_scale         The scale of the hidden state.
     *                                         input_to_input_weights     (Optional) 2D weights tensor with dimensions [input_size, num_units]. Data type supported: QSYMM8.
     *                                         recurrent_to_input_weights (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                         cell_to_input_weights      (Optional) 1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: QSYMM16.
     *                                         cell_to_forget_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_to_output_weights     (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         input_gate_bias            (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: S32.
     *                                         projection_weights         (Optional) 2D weights tensor with dimensions [output_size, num_units]. Data type supported: QSYMM8.
     *                                         projection_bias            (Optional) 1D weights tensor with dimensions [output_size]. S32.
     *                                         input_layer_norm_weights   (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         forget_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_layer_norm_weights    (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         output_layer_norm_weights  (Optional) 1D weights tensor with dimensions [num_units]. Data type supported: QSYMM16.
     *                                         cell_threshold             (Optional) The clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip].
     *                                                                               If set to 0.0 then clipping is disabled.
     *                                         projection_threshold       (Optional) The clipping threshold for the output from the projection layer, such that values are bound within
     *                                                                               [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     * @return a status
     */
    static Status validate(const ITensorInfo *input,
                           const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                           const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                           const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                           const ITensorInfo *cell_state_in, const ITensorInfo *output_state_in,
                           const ITensorInfo *cell_state_out, const ITensorInfo *output_state_out,
                           const LSTMParams<ITensorInfo> &lstm_params);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    enum class LayerNormGate : uint8_t
    {
        Forget,
        Cell,
        Input,
        Output,
        Count
    };
    static constexpr uint8_t _layer_norm_count = static_cast<uint8_t>(LayerNormGate::Count);

    /** Internal method to configure matrix multiplication plus output stage of each gate.
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in] mm              Matrix multiplication function to use.
     * @param[in] outstage        Output stage function to use.
     * @param[in] gemmlowp_info   GEMMLowp metadata to be used by the output stage.
     * @param[in] mm_input        Input tensor to matrix multiplication function.
     * @param[in] mm_weights      Weights tensor to matrix multiplication function.
     * @param[in] bias            Bias tensor to matrix multiplication function.
     * @param[in] outstage_res    Tensor to be used for storing the result of the output stage.
     * @param[in] gemmlowp_scale  Real multiplier to be used computing multiplier and shift for requantization.
     * @param[in] mm_res_info     Tensor info to be used to initialize matrix multiplication result tensor.
     * @param[in] mm_res_info     Tensor info to be used to initialize output stage result tensor.
     *
     */
    void configure_mm(const CLCompileContext &compile_context, CLGEMMLowpMatrixMultiplyCore &mm, CLGEMMLowpOutputStage &outstage, GEMMLowpOutputStageInfo &gemmlowp_info,
                      const ICLTensor *mm_input, const ICLTensor *mm_weights, const ICLTensor *bias, CLTensor *mm_res,
                      CLTensor *outstage_res, float gemmlowp_scale,
                      const TensorInfo &mm_res_info, const TensorInfo &outstage_tensor_info);

    MemoryGroup _memory_group{};

    // Functions used
    CLTranspose                          _transpose_input_to_forget_weights{};
    CLTranspose                          _transpose_input_to_cell_weights{};
    CLTranspose                          _transpose_input_to_output_weights{};
    CLTranspose                          _transpose_input_to_input_weights{};
    CLTranspose                          _transpose_recurrent_to_forget_weights{};
    CLTranspose                          _transpose_recurrent_to_cell_weights{};
    CLTranspose                          _transpose_recurrent_to_output_weights{};
    CLTranspose                          _transpose_recurrent_to_input_weights{};
    CLTranspose                          _transpose_projection_weights{};
    CLGEMMLowpMatrixAReductionKernel     _input_to_input_reduction{};
    CLGEMMLowpMatrixAReductionKernel     _recurrent_to_input_reduction{};
    CLGEMMLowpMatrixAReductionKernel     _input_to_forget_reduction{};
    CLGEMMLowpMatrixAReductionKernel     _recurrent_to_forget_reduction{};
    CLGEMMLowpMatrixAReductionKernel     _input_to_cell_reduction{};
    CLGEMMLowpMatrixAReductionKernel     _recurrent_to_cell_reduction{};
    CLGEMMLowpMatrixAReductionKernel     _input_to_output_reduction{};
    CLGEMMLowpMatrixAReductionKernel     _recurrent_to_output_reduction{};
    CLGEMMLowpMatrixAReductionKernel     _projection_reduction{};
    CLSaturatedArithmeticOperationKernel _projection_bias_add{};
    CLGEMMLowpMatrixMultiplyCore         _mm_input_to_forget{};
    CLGEMMLowpMatrixMultiplyCore         _mm_recurrent_to_forget{};
    CLPixelWiseMultiplicationKernel      _pixelwise_mul_cell_to_forget{};
    CLGEMMLowpOutputStage                _input_to_forget_outstage{};
    CLGEMMLowpOutputStage                _recurrent_to_forget_outstage{};
    CLGEMMLowpOutputStage                _cell_to_forget_outstage{};
    CLSaturatedArithmeticOperationKernel _accumulate_input_recurrent_forget{};
    CLSaturatedArithmeticOperationKernel _accumulate_cell_forget{};
    CLActivationLayer                    _forget_gate_sigmoid{};
    CLGEMMLowpMatrixMultiplyCore         _mm_input_to_cell{};
    CLGEMMLowpOutputStage                _input_to_cell_outstage{};
    CLGEMMLowpMatrixMultiplyCore         _mm_recurrent_to_cell{};
    CLGEMMLowpOutputStage                _recurrent_to_cell_outstage{};
    CLSaturatedArithmeticOperationKernel _accumulate_input_recurrent_modulation{};
    CLActivationLayer                    _cell_gate_tanh{};
    CLSaturatedArithmeticOperationKernel _input_gate_sub{};
    CLGEMMLowpMatrixMultiplyCore         _mm_input_to_input{};
    CLGEMMLowpOutputStage                _input_to_input_outstage{};
    CLGEMMLowpMatrixMultiplyCore         _mm_recurrent_to_input{};
    CLGEMMLowpOutputStage                _recurrent_to_input_outstage{};
    CLSaturatedArithmeticOperationKernel _accumulate_input_recurrent_input{};
    CLPixelWiseMultiplicationKernel      _pixelwise_mul_cell_to_input{};
    CLGEMMLowpOutputStage                _cell_to_input_outstage{};
    CLSaturatedArithmeticOperationKernel _accumulate_cell_input{};
    CLActivationLayer                    _input_gate_tanh{};
    CLPixelWiseMultiplicationKernel      _pixelwise_mul_forget_cell{};
    CLPixelWiseMultiplicationKernel      _pixelwise_mul_input_cell{};
    CLSaturatedArithmeticOperationKernel _add_forget_cell{};
    CLActivationLayer                    _cell_clip{};
    CLGEMMLowpMatrixMultiplyCore         _mm_input_to_output{};
    CLGEMMLowpOutputStage                _input_to_output_outstage{};
    CLGEMMLowpMatrixMultiplyCore         _mm_recurrent_to_output{};
    CLGEMMLowpOutputStage                _recurrent_to_output_outstage{};
    CLSaturatedArithmeticOperationKernel _accumulate_input_recurrent_output{};
    CLPixelWiseMultiplicationKernel      _pixelwise_mul_cell_to_output{};
    CLSaturatedArithmeticOperationKernel _accumulate_cell_to_output{};
    CLActivationLayer                    _output_gate_sigmoid{};
    CLActivationLayer                    _hidden_tanh{};
    CLPixelWiseMultiplicationKernel      _pixelwise_mul_hidden{};
    CLGEMMLowpOutputStage                _hidden_outstage{};
    CLGEMMLowpMatrixMultiplyCore         _mm_projection{};
    CLGEMMLowpOutputStage                _projection_outstage{};
    CLSaturatedArithmeticOperationKernel _accumulate_projection{};
    CLActivationLayer                    _projection_clip{};
    std::array<CLQLSTMLayerNormalizationKernel, _layer_norm_count> _layer_norms{ {} };

    // Tensor pointers
    const ICLTensor *_input_to_input_weights
    {
        nullptr
    };
    const ICLTensor *_recurrent_to_input_weights{ nullptr };
    const ICLTensor *_projection_bias{ nullptr };
    const ICLTensor *_input_to_forget_weights{ nullptr };
    const ICLTensor *_input_to_cell_weights{ nullptr };
    const ICLTensor *_input_to_output_weights{ nullptr };
    const ICLTensor *_recurrent_to_forget_weights{ nullptr };
    const ICLTensor *_recurrent_to_cell_weights{ nullptr };
    const ICLTensor *_recurrent_to_output_weights{ nullptr };
    const ICLTensor *_projection_weights{ nullptr };
    std::array<const ICLTensor *, _layer_norm_count> _layer_norm_weights{ {} };
    std::array<const ICLTensor *, _layer_norm_count> _layer_norm_bias{ {} };

    using LayerNormIndexType = typename std::underlying_type<LayerNormGate>::type;
    inline LayerNormIndexType getGateIndex(LayerNormGate g)
    {
        return static_cast<LayerNormIndexType>(g);
    }

    inline void set_layer_norm_weight(const ICLTensor *t, LayerNormGate g)
    {
        _layer_norm_weights[getGateIndex(g)] = t;
    }

    inline void set_layer_norm_bias(const ICLTensor *t, LayerNormGate g)
    {
        _layer_norm_bias[getGateIndex(g)] = t;
    }

    inline const ICLTensor *get_layer_norm_weight(LayerNormGate g)
    {
        return _layer_norm_weights[getGateIndex(g)];
    }

    inline const ICLTensor *get_layer_norm_bias(LayerNormGate g)
    {
        return _layer_norm_bias[getGateIndex(g)];
    }

    inline CLQLSTMLayerNormalizationKernel &get_layer_norm(LayerNormGate g)
    {
        return _layer_norms[getGateIndex(g)];
    }

    inline void configure_layer_norm(LayerNormGate g, const ICLTensor *in)
    {
        ARM_COMPUTE_ERROR_ON(!_has_layer_norm);

        CLTensor *out = &get_layer_norm_output(g);
        _memory_group.manage(out);
        out->allocator()->init(*(in->info()));

        get_layer_norm(g).configure(in, out, get_layer_norm_weight(g), get_layer_norm_bias(g));
    }

    inline static Status validate_layer_norm(const ITensorInfo &in, const ITensorInfo &weight, const ITensorInfo &bias)
    {
        // Output quantization scale will be different, but ignored here
        // since it will be configured at configure() stage.
        const TensorInfo out
        {
            in
        };
        return CLQLSTMLayerNormalizationKernel::validate(&in, &out, &weight, &bias);
    }

    // Temporary tensors
    CLTensor _input_to_forget_weights_transposed{ nullptr };
    CLTensor _input_to_cell_weights_transposed{ nullptr };
    CLTensor _input_to_output_weights_transposed{ nullptr };
    CLTensor _input_to_input_weights_transposed{ nullptr };
    CLTensor _recurrent_to_forget_weights_transposed{ nullptr };
    CLTensor _recurrent_to_cell_weights_transposed{ nullptr };
    CLTensor _recurrent_to_output_weights_transposed{ nullptr };
    CLTensor _recurrent_to_input_weights_transposed{ nullptr };
    CLTensor _projection_weights_transposed{ nullptr };
    CLTensor _input_to_input_eff_bias{ nullptr };
    CLTensor _recurrent_to_input_eff_bias{ nullptr };
    CLTensor _input_to_forget_eff_bias{ nullptr };
    CLTensor _recurrent_to_forget_eff_bias{ nullptr };
    CLTensor _input_to_cell_eff_bias{ nullptr };
    CLTensor _recurrent_to_cell_eff_bias{ nullptr };
    CLTensor _input_to_output_eff_bias{ nullptr };
    CLTensor _recurrent_to_output_eff_bias{ nullptr };
    CLTensor _projection_reduction_res{ nullptr };
    CLTensor _projection_eff_bias{ nullptr };
    CLTensor _mm_input_to_forget_res{ nullptr };
    CLTensor _mm_recurrent_to_forget_res{ nullptr };
    CLTensor _mul_cell_to_forget_res{ nullptr };
    CLTensor _input_to_forget_outstage_res{ nullptr };
    CLTensor _cell_to_forget_outstage_res{ nullptr };
    CLTensor _recurrent_to_forget_outstage_res{ nullptr };
    CLTensor _forget_gate{ nullptr };
    CLTensor _mm_input_to_cell_res{ nullptr };
    CLTensor _input_to_cell_outstage_res{ nullptr };
    CLTensor _mm_recurrent_to_cell_res{ nullptr };
    CLTensor _recurrent_to_cell_outstage_res{ nullptr };
    CLTensor _cell_gate{ nullptr };
    CLTensor _mul_input_cell_res{ nullptr };
    CLTensor _mm_input_to_input_res{ nullptr };
    CLTensor _input_to_input_outstage_res{ nullptr };
    CLTensor _mm_recurrent_to_input_res{ nullptr };
    CLTensor _mul_cell_to_input_res{ nullptr };
    CLTensor _cell_to_input_outstage_res{ nullptr };
    CLTensor _recurrent_to_input_outstage_res{ nullptr };
    CLTensor _input_gate{ nullptr };
    CLTensor _mm_input_to_output_res{ nullptr };
    CLTensor _input_to_output_outstage_res{ nullptr };
    CLTensor _mm_recurrent_to_output_res{ nullptr };
    CLTensor _mul_cell_to_output_res{ nullptr };
    CLTensor _recurrent_to_output_outstage_res{ nullptr };
    CLTensor _output_gate{ nullptr };
    CLTensor _hidden_mul_res{ nullptr };
    CLTensor _mm_projection_res{ nullptr };
    CLTensor _projection_outstage_res{ nullptr };
    CLTensor _ones{ nullptr };
    std::array<CLTensor, _layer_norm_count> _layer_norm_output{ {} };

    inline CLTensor &get_layer_norm_output(LayerNormGate g)
    {
        return _layer_norm_output[getGateIndex(g)];
    }

    bool _is_prepared{ false };
    bool _has_cifg{ false };
    bool _has_cell_clipping{ false };
    bool _has_projection{ false };
    bool _has_projection_clipping{ false };
    bool _has_peephole{ false };
    bool _has_layer_norm{ false };
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLQLSTMLAYER_H */
