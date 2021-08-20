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
#ifndef ARM_COMPUTE_CLLSTMLAYERQUANTIZED_H
#define ARM_COMPUTE_CLLSTMLAYERQUANTIZED_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLConcatenateLayer.h"
#include "arm_compute/runtime/CL/functions/CLDequantizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h"
#include "arm_compute/runtime/CL/functions/CLQuantizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLSlice.h"
#include "arm_compute/runtime/CL/functions/CLTranspose.h"

#include "arm_compute/runtime/common/LSTMParams.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Basic function to run @ref CLLSTMLayerQuantized
 *
 * This function calls the following CL functions/kernels:
 *
 * -# @ref CLGEMMLowpMatrixMultiplyCore      Quantized matrix multiplication core. Accumulators are 32-bit integers
 * -# @ref CLGEMMLowpOutputStage             Convert 32-bit integers into QSYMM16
 * -# @ref CLTranspose                       Matrix transpose
 * -# @ref CLConcatenateLayer                Tensor concatenation
 * -# @ref CLActivationLayer                 Activation functions (tanh and logistic)
 * -# @ref CLArithmeticAddition              Elementwise addition
 * -# @ref CLPixelWiseMultiplication         Elementwise multiplication
 * -# @ref CLSlice                           Tensor slicing
 * -# @ref CLDequantizationLayer             Dequantize into float
 * -# @ref CLQuantizationLayer               Quantize from float
 * */
class CLLSTMLayerQuantized : public IFunction
{
public:
    /** Default constructor */
    CLLSTMLayerQuantized(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLSTMLayerQuantized(const CLLSTMLayerQuantized &) = delete;
    /** Default move constructor */
    CLLSTMLayerQuantized(CLLSTMLayerQuantized &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLSTMLayerQuantized &operator=(const CLLSTMLayerQuantized &) = delete;
    /** Default move assignment operator */
    CLLSTMLayerQuantized &operator=(CLLSTMLayerQuantized &&) = default;
    /** Initialize function's tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0 - src8 |src9 - src12 |src13   |src14  |dst0   |dst1   |
     * |:-----------|:------------|:-------|:------|:------|:------|
     * |QASYMM8     |S32          |QSYMM16 |QASYMM8|QSYMM16|QASYMM8|
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
    void configure(const ICLTensor *input,
                   const ICLTensor *input_to_input_weights, const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                   const ICLTensor *recurrent_to_input_weights, const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                   const ICLTensor *input_gate_bias, const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                   ICLTensor *cell_state_in, const ICLTensor *output_state_in,
                   ICLTensor *cell_state_out, ICLTensor *output_state_out);
    /** Initialize function's tensors.
     *
     * @param[in]  compile_context             The compile context to be used.
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
    void configure(const CLCompileContext &compile_context, const ICLTensor *input,
                   const ICLTensor *input_to_input_weights, const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                   const ICLTensor *recurrent_to_input_weights, const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                   const ICLTensor *input_gate_bias, const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                   ICLTensor *cell_state_in, const ICLTensor *output_state_in,
                   ICLTensor *cell_state_out, ICLTensor *output_state_out);

    /** Static function to check if given info will lead to a valid configuration of @ref CLLSTMLayerQuantized
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
    CLGEMMLowpMatrixMultiplyCore _gemmlowp;
    CLGEMMLowpOutputStage        _output_stage;
    CLTranspose                  _transpose_weights;
    CLConcatenateLayer           _concat_input_weights;
    CLConcatenateLayer           _concat_recurrent_weights;
    CLConcatenateLayer           _concat_weights;
    CLConcatenateLayer           _concat_inputs;
    CLConcatenateLayer           _concat_bias;
    CLActivationLayer            _sigmoid_forget_gate;
    CLActivationLayer            _sigmoid_input_gate;
    CLActivationLayer            _sigmoid_output_gate;
    CLActivationLayer            _tanh_modulation_gate;
    CLActivationLayer            _tanh_output_state;
    CLArithmeticAddition         _add_cell_state_tmps;
    CLArithmeticAddition         _add2;
    CLPixelWiseMultiplication    _mul_forget_gate_cell_state;
    CLPixelWiseMultiplication    _mul_input_gate_input_mod_gate;
    CLPixelWiseMultiplication    _mul_output_state_tmp_output_gate;
    CLSlice                      _slice_input_tensor;
    CLSlice                      _slice_forget_tensor;
    CLSlice                      _slice_cell_tensor;
    CLSlice                      _slice_output_tensor;
    CLDequantizationLayer        _dequantize;
    CLQuantizationLayer          _quantize;

    // Tensor pointers
    const ICLTensor *_input_to_input_weights;
    const ICLTensor *_input_to_forget_weights;
    const ICLTensor *_input_to_cell_weights;
    const ICLTensor *_input_to_output_weights;
    const ICLTensor *_recurrent_to_input_weights;
    const ICLTensor *_recurrent_to_forget_weights;
    const ICLTensor *_recurrent_to_cell_weights;
    const ICLTensor *_recurrent_to_output_weights;
    const ICLTensor *_input_gate_bias;
    const ICLTensor *_forget_gate_bias;
    const ICLTensor *_cell_bias;
    const ICLTensor *_output_gate_bias;

    // Temporary tensors
    CLTensor _recurrent_weights;
    CLTensor _input_weights;
    CLTensor _weights;
    CLTensor _input;
    CLTensor _weights_transposed;
    CLTensor _output_highp;
    CLTensor _output_lowp;
    CLTensor _bias;
    CLTensor _forget_gate_input;
    CLTensor _input_gate_input;
    CLTensor _output_gate_input;
    CLTensor _input_modulation_gate_input;
    CLTensor _forget_gate_output;
    CLTensor _input_gate_output;
    CLTensor _output_gate_output;
    CLTensor _input_modulation_gate_output;
    CLTensor _cell_state_tmp1;
    CLTensor _cell_state_tmp2;
    CLTensor _output_state_tmp;
    CLTensor _output_state_out_symm;
    CLTensor _output_state_out_f32;

    bool _is_prepared;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLLSTMLAYERQUANTIZED_H */
