/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NELSTMLAYERQUANTIZED_H
#define ARM_COMPUTE_NELSTMLAYERQUANTIZED_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDequantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include "arm_compute/runtime/NEON/functions/NEQuantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NESlice.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"

#include "arm_compute/runtime/common/LSTMParams.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to run @ref NELSTMLayerQuantized
 *
 * This function calls the following Neon functions/kernels:
 *
 * -# @ref NEGEMMLowpMatrixMultiplyCore                          Quantized matrix multiplication core. Accumulators are 32-bit integers
 * -# @ref NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint   Convert 32-bit integers into QSYMM16
 * -# @ref NETranspose                                           Matrix transpose
 * -# @ref NEConcatenateLayer                                    Tensor concatenation
 * -# @ref NEActivationLayer                                     Activation functions (tanh and logistic)
 * -# @ref NEArithmeticAddition                                  Elementwise addition
 * -# @ref NEPixelWiseMultiplication                             Elementwise multiplication
 * -# @ref NESlice                                               Tensor slicing
 * -# @ref NEDequantizationLayer                                 Dequantize into float
 * -# @ref NEQuantizationLayer                                   Quantize from float
 * */
class NELSTMLayerQuantized : public IFunction
{
public:
    /** Default constructor */
    NELSTMLayerQuantized(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELSTMLayerQuantized(const NELSTMLayerQuantized &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NELSTMLayerQuantized(NELSTMLayerQuantized &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELSTMLayerQuantized &operator=(const NELSTMLayerQuantized &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NELSTMLayerQuantized &operator=(NELSTMLayerQuantized &&) = delete;
    /** Default destructor */
    ~NELSTMLayerQuantized();
    /** Initialize function's tensors.
     *
     * @param[in]  input                       Source tensor. Input is a 2D tensor with dimensions [input_size, batch_size]. Data types supported: QASYMM8.
     * @param[in]  input_to_input_weights      2D weights tensor with dimensions [input_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  input_to_forget_weights     2D weights tensor with dimensions [input_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  input_to_cell_weights       2D weights tensor with dimensions [input_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  input_to_output_weights     2D weights tensor with dimensions [input_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_input_weights  2D weights tensor with dimensions [output_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_forget_weights 2D weights tensor with dimensions [output_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_cell_weights   2D weights tensor with dimensions [output_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_output_weights 2D weights tensor with dimensions [output_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  input_gate_bias             1D weights tensor with dimensions [output_size]. Data type supported: S32.
     * @param[in]  forget_gate_bias            1D weights tensor with dimensions [output_size]. Data type supported: S32.
     * @param[in]  cell_bias                   1D weights tensor with dimensions [output_size]. Data type supported: S32.
     * @param[in]  output_gate_bias            1D weights tensor with dimensions [output_size]. Data type supported: S32.
     * @param[in]  cell_state_in               2D tensor with dimensions [output_size, batch_size]. Data type supported:  QSYMM16.
     * @param[in]  output_state_in             2D tensor with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[out] cell_state_out              Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size]. Data type supported:  QSYMM16.
     * @param[out] output_state_out            Destination tensor. Output is a 2D tensor with dimensions [output_size, batch_size].Data types supported: Same as @p input.
     */
    void configure(const ITensor *input,
                   const ITensor *input_to_input_weights, const ITensor *input_to_forget_weights, const ITensor *input_to_cell_weights, const ITensor *input_to_output_weights,
                   const ITensor *recurrent_to_input_weights, const ITensor *recurrent_to_forget_weights, const ITensor *recurrent_to_cell_weights, const ITensor *recurrent_to_output_weights,
                   const ITensor *input_gate_bias, const ITensor *forget_gate_bias, const ITensor *cell_bias, const ITensor *output_gate_bias,
                   ITensor *cell_state_in, const ITensor *output_state_in,
                   ITensor *cell_state_out, ITensor *output_state_out);

    /** Static function to check if given info will lead to a valid configuration of @ref NELSTMLayer
     *
     * @param[in]  input                       Source tensor info. Input is a 2D tensor info with dimensions [input_size, batch_size]. Data types supported: QASYMM8.
     * @param[in]  input_to_input_weights      2D weights tensor info with dimensions [input_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  input_to_forget_weights     2D weights tensor info with dimensions [input_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  input_to_cell_weights       2D weights tensor info with dimensions [input_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  input_to_output_weights     2D weights tensor info with dimensions [input_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_input_weights  2D weights tensor info with dimensions [output_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_forget_weights 2D weights tensor info with dimensions [output_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_cell_weights   2D weights tensor info with dimensions [output_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  recurrent_to_output_weights 2D weights tensor info with dimensions [output_size, output_size]. Data type supported: Same as @p input.
     * @param[in]  input_gate_bias             1D weights tensor info with dimensions [output_size]. Data type supported: S32.
     * @param[in]  forget_gate_bias            1D weights tensor info with dimensions [output_size]. Data type supported: S32.
     * @param[in]  cell_bias                   1D weights tensor info with dimensions [output_size]. Data type supported: S32.
     * @param[in]  output_gate_bias            1D weights tensor info with dimensions [output_size]. Data type supported: S32.
     * @param[in]  cell_state_in               2D tensor info with dimensions [output_size, batch_size]. Data type supported:  QSYMM16.
     * @param[in]  output_state_in             2D tensor info with dimensions [output_size, batch_size]. Data type supported: Same as @p input.
     * @param[out] cell_state_out              Destination tensor info. Output is a 2D tensor info with dimensions [output_size, batch_size]. Data type supported:  QSYMM16.
     * @param[out] output_state_out            Destination tensor info. Output is a 2D tensor info with dimensions [output_size, batch_size].Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input,
                           const ITensorInfo *input_to_input_weights, const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                           const ITensorInfo *recurrent_to_input_weights, const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                           const ITensorInfo *input_gate_bias, const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                           const ITensorInfo *cell_state_in, const ITensorInfo *output_state_in,
                           const ITensorInfo *cell_state_out, const ITensorInfo *output_state_out);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup _memory_group;

    // Functions used
    NEGEMMLowpMatrixMultiplyCore                        _gemmlowp;
    NEGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint _output_stage;
    NETranspose                                         _transpose_weights;
    NEConcatenateLayer                                  _concat_input_weights;
    NEConcatenateLayer                                  _concat_recurrent_weights;
    NEConcatenateLayer                                  _concat_weights;
    NEConcatenateLayer                                  _concat_inputs;
    NEConcatenateLayer                                  _concat_bias;
    NEActivationLayer                                   _sigmoid_forget_gate;
    NEActivationLayer                                   _sigmoid_input_gate;
    NEActivationLayer                                   _sigmoid_output_gate;
    NEActivationLayer                                   _tanh_modulation_gate;
    NEActivationLayer                                   _tanh_output_state;
    NEArithmeticAddition                                _add1;
    NEArithmeticAddition                                _add2;
    NEPixelWiseMultiplication                           _mul1;
    NEPixelWiseMultiplication                           _mul2;
    NEPixelWiseMultiplication                           _mul3;
    NESlice                                             _slice_input_tensor;
    NESlice                                             _slice_forget_tensor;
    NESlice                                             _slice_cell_tensor;
    NESlice                                             _slice_output_tensor;
    NEDequantizationLayer                               _dequantize;
    NEQuantizationLayer                                 _quantize;

    // Tensor pointers
    const ITensor *_input_to_input_weights;
    const ITensor *_input_to_forget_weights;
    const ITensor *_input_to_cell_weights;
    const ITensor *_input_to_output_weights;
    const ITensor *_recurrent_to_input_weights;
    const ITensor *_recurrent_to_forget_weights;
    const ITensor *_recurrent_to_cell_weights;
    const ITensor *_recurrent_to_output_weights;
    const ITensor *_input_gate_bias;
    const ITensor *_forget_gate_bias;
    const ITensor *_cell_bias;
    const ITensor *_output_gate_bias;

    // Temporary tensors
    Tensor _recurrent_weights;
    Tensor _input_weights;
    Tensor _weights;
    Tensor _input;
    Tensor _weights_transposed;
    Tensor _output_highp;
    Tensor _output_lowp;
    Tensor _bias;
    Tensor _forget_gate_input;
    Tensor _input_gate_input;
    Tensor _output_gate_input;
    Tensor _input_modulation_gate_input;
    Tensor _forget_gate_output;
    Tensor _input_gate_output;
    Tensor _output_gate_output;
    Tensor _input_modulation_gate_output;
    Tensor _cell_state1;
    Tensor _cell_state2;
    Tensor _output_state_tmp;
    Tensor _output_state_out_symm;
    Tensor _output_state_out_f32;

    bool _is_prepared;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NELSTMLAYERQUANTIZED_H */
