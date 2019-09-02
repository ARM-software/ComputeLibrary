/*
 * Copyright (c) 2019 ARM Limited.
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

#include "arm_compute/runtime/CL/functions/CLLSTMLayerQuantized.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include <cmath>
#include <memory>
#include <tuple>

namespace arm_compute
{
namespace
{
// Quantization info structures used in the LSTMQuantize layer
const QuantizationInfo qasymm(1.f / 128.f, 128);
const QuantizationInfo qsymm_3(8.f / 32768.f, 0);  // qsymm16 with 3 integer bit
const QuantizationInfo qsymm_4(16.f / 32768.f, 0); // qsymm16 with 4 integer bit
const QuantizationInfo qsymm_0(1.f / 32768.f, 0);  // qsymm16 with 0 integer bit
} // namespace

CLLSTMLayerQuantized::CLLSTMLayerQuantized(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _gemmlowp(), _output_stage(), _transpose_weights(), _concat_input_weights(), _concat_recurrent_weights(), _concat_weights(), _concat_inputs(),
      _concat_bias(), _sigmoid_forget_gate(), _sigmoid_input_gate(), _sigmoid_output_gate(), _tanh_modulation_gate(), _tanh_output_state(), _add_cell_state_tmps(), _add2(), _mul_forget_gate_cell_state(),
      _mul_input_gate_input_mod_gate(), _mul_output_state_tmp_output_gate(), _slice_input_tensor(), _slice_forget_tensor(), _slice_cell_tensor(), _slice_output_tensor(), _dequantize(), _quantize(),
      _input_to_input_weights(nullptr), _input_to_forget_weights(nullptr), _input_to_cell_weights(nullptr), _input_to_output_weights(nullptr), _recurrent_to_input_weights(nullptr),
      _recurrent_to_forget_weights(nullptr), _recurrent_to_cell_weights(nullptr), _recurrent_to_output_weights(nullptr), _input_gate_bias(nullptr), _forget_gate_bias(nullptr), _cell_bias(nullptr),
      _output_gate_bias(nullptr), _recurrent_weights(), _input_weights(), _weights(), _input(), _weights_transposed(), _output_highp(), _output_lowp(), _bias(), _forget_gate_input(), _input_gate_input(),
      _output_gate_input(), _input_modulation_gate_input(), _forget_gate_output(), _input_gate_output(), _output_gate_output(), _input_modulation_gate_output(), _cell_state_tmp1(), _cell_state_tmp2(),
      _output_state_tmp(), _output_state_out_symm(), _output_state_out_f32(), _is_prepared(false)
{
}

void CLLSTMLayerQuantized::configure(const ICLTensor *input,
                                     const ICLTensor *input_to_input_weights, const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                                     const ICLTensor *recurrent_to_input_weights, const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                                     const ICLTensor *input_gate_bias, const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                                     ICLTensor *cell_state_in, const ICLTensor *output_state_in,
                                     ICLTensor *cell_state_out, ICLTensor *output_state_out)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights,
                                 recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights,
                                 input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias, cell_state_in, output_state_in, cell_state_out, output_state_out);

    ARM_COMPUTE_ERROR_THROW_ON(CLLSTMLayerQuantized::validate(input->info(), input_to_input_weights->info(), input_to_forget_weights->info(), input_to_cell_weights->info(),
                                                              input_to_output_weights->info(),
                                                              recurrent_to_input_weights->info(), recurrent_to_forget_weights->info(), recurrent_to_cell_weights->info(), recurrent_to_output_weights->info(),
                                                              input_gate_bias->info(), forget_gate_bias->info(), cell_bias->info(), output_gate_bias->info(), cell_state_in->info(), output_state_in->info(), cell_state_out->info(), output_state_out->info()));

    const int input_size  = input->info()->dimension(0);
    const int batch_size  = input->info()->dimension(1);
    const int output_size = input_to_input_weights->info()->dimension(1);

    const QuantizationInfo qweights = input_to_input_weights->info()->quantization_info(); // Weights quantization

    auto_init_if_empty(*cell_state_out->info(), TensorInfo(TensorShape(batch_size, output_size), 1, DataType::QSYMM16, qsymm_4));
    auto_init_if_empty(*output_state_out->info(), TensorInfo(TensorShape(batch_size, output_size), 1, DataType::QASYMM8, qasymm));

    _input_to_input_weights      = input_to_input_weights;
    _input_to_forget_weights     = input_to_forget_weights;
    _input_to_cell_weights       = input_to_cell_weights;
    _input_to_output_weights     = input_to_output_weights;
    _recurrent_to_input_weights  = recurrent_to_input_weights;
    _recurrent_to_forget_weights = recurrent_to_forget_weights;
    _recurrent_to_cell_weights   = recurrent_to_cell_weights;
    _recurrent_to_output_weights = recurrent_to_output_weights;
    _input_gate_bias             = input_gate_bias;
    _forget_gate_bias            = forget_gate_bias;
    _cell_bias                   = cell_bias;
    _output_gate_bias            = output_gate_bias;

    // Weights concatenation
    std::vector<const ICLTensor *> inputs_weights_vector;
    inputs_weights_vector.emplace_back(input_to_input_weights);
    inputs_weights_vector.emplace_back(input_to_forget_weights);
    inputs_weights_vector.emplace_back(input_to_cell_weights);
    inputs_weights_vector.emplace_back(input_to_output_weights);

    std::vector<const ICLTensor *> recurrent_weights_vector;
    recurrent_weights_vector.emplace_back(recurrent_to_input_weights);
    recurrent_weights_vector.emplace_back(recurrent_to_forget_weights);
    recurrent_weights_vector.emplace_back(recurrent_to_cell_weights);
    recurrent_weights_vector.emplace_back(recurrent_to_output_weights);

    _input_weights.allocator()->init(TensorInfo(TensorShape(input_size, 4 * output_size), 1, DataType::QASYMM8, qweights));
    _concat_input_weights.configure(inputs_weights_vector, &_input_weights, Window::DimY);

    _recurrent_weights.allocator()->init(TensorInfo(TensorShape(output_size, 4 * output_size), 1, DataType::QASYMM8, qweights));
    _concat_recurrent_weights.configure(recurrent_weights_vector, &_recurrent_weights, Window::DimY);

    std::vector<const ICLTensor *> weights_vector;
    weights_vector.emplace_back(&_recurrent_weights);
    weights_vector.emplace_back(&_input_weights);

    _weights.allocator()->init(TensorInfo(TensorShape(output_size + input_size, 4 * output_size), 1, DataType::QASYMM8, qweights));
    _concat_weights.configure(weights_vector, &_weights, Window::DimX);
    _transpose_weights.configure(&_weights, &_weights_transposed);

    // Input concatenation
    std::vector<const ICLTensor *> input_vector;
    input_vector.emplace_back(input);
    input_vector.emplace_back(output_state_in);

    _memory_group.manage(&_input);
    _input.allocator()->init(TensorInfo(TensorShape(output_size + input_size, batch_size), 1, DataType::QASYMM8, qasymm));
    _concat_inputs.configure(input_vector, &_input, Window::DimX);

    // Bias concatenation
    std::vector<const ICLTensor *> bias_vector;
    bias_vector.emplace_back(input_gate_bias);
    bias_vector.emplace_back(forget_gate_bias);
    bias_vector.emplace_back(cell_bias);
    bias_vector.emplace_back(output_gate_bias);

    _bias.allocator()->init(TensorInfo(TensorShape(4 * output_size), 1, DataType::S32));
    _concat_bias.configure(bias_vector, &_bias, Window::DimX);

    // Invert the offset for gemmlowp
    _input.info()->set_quantization_info(QuantizationInfo(qasymm.uniform().scale, -qasymm.uniform().offset));
    _weights_transposed.info()->set_quantization_info(QuantizationInfo(qweights.uniform().scale, -qweights.uniform().offset));

    // Run gemmlowp
    _memory_group.manage(&_output_highp);
    _output_highp.allocator()->init(TensorInfo(TensorShape(4 * output_size, batch_size), 1, DataType::S32));
    _gemmlowp.configure(&_input, &_weights_transposed, nullptr, &_output_highp);
    _input.allocator()->allocate();

    // Set the offset back
    _input.info()->set_quantization_info(QuantizationInfo(qasymm.uniform().scale, qasymm.uniform().offset));
    _weights_transposed.info()->set_quantization_info(QuantizationInfo(qweights.uniform().scale, qweights.uniform().offset));

    // multiplier = (input_scale * weights_scale) / output_scale (2 ^ (-12))
    _output_lowp.allocator()->init(TensorInfo(_output_highp.info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_3));

    const float multiplier        = 4096.f * qasymm.uniform().scale * qweights.uniform().scale;
    int         output_multiplier = 0;
    int         output_shift      = 0;

    quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

    _memory_group.manage(&_output_lowp);
    _output_stage.configure(&_output_highp, &_bias, &_output_lowp, output_multiplier, output_shift);
    _output_highp.allocator()->allocate();
    _bias.allocator()->allocate();

    // Get the gate tensors
    if(batch_size > 1)
    {
        _memory_group.manage(&_input_gate_input);
        _slice_input_tensor.configure(&_output_lowp, &_input_gate_input, { 0, 0 }, { output_size, batch_size });
        _memory_group.manage(&_forget_gate_input);
        _slice_forget_tensor.configure(&_output_lowp, &_forget_gate_input, { output_size, 0 }, { 2 * output_size, batch_size });
        _memory_group.manage(&_input_modulation_gate_input);
        _slice_cell_tensor.configure(&_output_lowp, &_input_modulation_gate_input, { 2 * output_size, 0 }, { 3 * output_size, batch_size });
        _memory_group.manage(&_output_gate_input);
        _slice_output_tensor.configure(&_output_lowp, &_output_gate_input, { 3 * output_size, 0 }, { 4 * output_size, batch_size });
        _output_lowp.allocator()->allocate();
    }
    else
    {
        _memory_group.manage(&_input_gate_input);
        _slice_input_tensor.configure(&_output_lowp, &_input_gate_input, { 0 }, { output_size });
        _memory_group.manage(&_forget_gate_input);
        _slice_forget_tensor.configure(&_output_lowp, &_forget_gate_input, { output_size }, { 2 * output_size });
        _memory_group.manage(&_input_modulation_gate_input);
        _slice_cell_tensor.configure(&_output_lowp, &_input_modulation_gate_input, { 2 * output_size }, { 3 * output_size });
        _memory_group.manage(&_output_gate_input);
        _slice_output_tensor.configure(&_output_lowp, &_output_gate_input, { 3 * output_size }, { 4 * output_size });
        _output_lowp.allocator()->allocate();
    }

    // Forget gate
    _memory_group.manage(&_forget_gate_output);
    _forget_gate_output.allocator()->init(TensorInfo(_forget_gate_input.info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_0));
    _sigmoid_forget_gate.configure(&_forget_gate_input, &_forget_gate_output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    _forget_gate_input.allocator()->allocate();

    // Input gate
    _memory_group.manage(&_input_gate_output);
    _input_gate_output.allocator()->init(TensorInfo(_input_gate_input.info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_0));
    _sigmoid_input_gate.configure(&_input_gate_input, &_input_gate_output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    _input_gate_input.allocator()->allocate();

    // Input modulation gate equation
    _memory_group.manage(&_input_modulation_gate_output);
    _input_modulation_gate_output.allocator()->init(TensorInfo(_input_modulation_gate_input.info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_0));
    _tanh_modulation_gate.configure(&_input_modulation_gate_input, &_input_modulation_gate_output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f));
    _input_modulation_gate_input.allocator()->allocate();

    // Output gate
    _memory_group.manage(&_output_gate_output);
    _output_gate_output.allocator()->init(TensorInfo(_output_gate_input.info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_0));
    _sigmoid_output_gate.configure(&_output_gate_input, &_output_gate_output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    _output_gate_input.allocator()->allocate();

    // Long term memory
    _memory_group.manage(&_cell_state_tmp1);
    _cell_state_tmp1.allocator()->init(TensorInfo(_forget_gate_output.info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_4));
    _mul_forget_gate_cell_state.configure(&_forget_gate_output, cell_state_in, &_cell_state_tmp1, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
    _forget_gate_output.allocator()->allocate();

    _memory_group.manage(&_cell_state_tmp2);
    _cell_state_tmp2.allocator()->init(TensorInfo(_input_gate_output.info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_4));
    _mul_input_gate_input_mod_gate.configure(&_input_gate_output, &_input_modulation_gate_output, &_cell_state_tmp2, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
    _input_modulation_gate_output.allocator()->allocate();
    _input_gate_output.allocator()->allocate();

    _add_cell_state_tmps.configure(&_cell_state_tmp1, &_cell_state_tmp2, cell_state_out, ConvertPolicy::SATURATE);
    _cell_state_tmp1.allocator()->allocate();
    _cell_state_tmp2.allocator()->allocate();

    // Short term memory
    _memory_group.manage(&_output_state_tmp);
    _output_state_tmp.allocator()->init(TensorInfo(cell_state_out->info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_0));
    _tanh_output_state.configure(cell_state_out, &_output_state_tmp, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f));

    _memory_group.manage(&_output_state_out_symm);
    _output_state_out_symm.allocator()->init(TensorInfo(_output_gate_output.info()->tensor_shape(), 1, DataType::QSYMM16, qsymm_0));
    _mul_output_state_tmp_output_gate.configure(&_output_state_tmp, &_output_gate_output, &_output_state_out_symm, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
    _output_gate_output.allocator()->allocate();
    _output_state_tmp.allocator()->allocate();

    // Requantize the output state from QSYMM16 to QASYMM8
    _memory_group.manage(&_output_state_out_f32);
    _output_state_out_f32.allocator()->init(TensorInfo(_output_state_out_symm.info()->tensor_shape(), 1, DataType::F32));
    _dequantize.configure(&_output_state_out_symm, &_output_state_out_f32);
    _output_state_out_symm.allocator()->allocate();

    _quantize.configure(&_output_state_out_f32, output_state_out);
    _output_state_out_f32.allocator()->allocate();
}

Status CLLSTMLayerQuantized::validate(const ITensorInfo *input,
                                      const ITensorInfo *input_to_input_weights, const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                                      const ITensorInfo *recurrent_to_input_weights, const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                                      const ITensorInfo *input_gate_bias, const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                                      const ITensorInfo *cell_state_in, const ITensorInfo *output_state_in,
                                      const ITensorInfo *cell_state_out, const ITensorInfo *output_state_out)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, recurrent_to_input_weights,
                                        recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights, input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias, cell_state_in,
                                        output_state_in, cell_state_out, output_state_out);

    const int input_size  = input->dimension(0);
    const int batch_size  = input->dimension(1);
    const int output_size = input_to_input_weights->dimension(1);

    // Dimensionality checks
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_to_input_weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_gate_bias->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(output_state_in->num_dimensions() > 2);

    TensorInfo input_weights_info(input_to_input_weights->clone()->set_tensor_shape(TensorShape(input_size, output_size)).set_data_type(DataType::QASYMM8));
    TensorInfo recurrent_weights_info(input_to_input_weights->clone()->set_tensor_shape(TensorShape(output_size, output_size)).set_data_type(DataType::QASYMM8));
    TensorInfo bias_info(input_gate_bias->clone()->set_tensor_shape(TensorShape(output_size)).set_data_type(DataType::S32));
    TensorInfo output_state_info(cell_state_in->clone()->set_tensor_shape(TensorShape(output_size, batch_size)).set_data_type(DataType::QASYMM8).set_quantization_info(qasymm));
    TensorInfo cell_state_info(cell_state_in->clone()->set_tensor_shape(TensorShape(output_size, batch_size)).set_data_type(DataType::QSYMM16).set_quantization_info(qsymm_4));

    // Shape checks
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&input_weights_info, input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&recurrent_weights_info, recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&bias_info, input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&cell_state_info, cell_state_in);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&output_state_info, output_state_in);

    // Data type checks
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input_weights_info, input, input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&recurrent_weights_info, recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&bias_info, input_gate_bias, forget_gate_bias, cell_bias, output_gate_bias);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&cell_state_info, cell_state_in);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&output_state_info, output_state_in);

    // Quantization checks
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&cell_state_info, cell_state_in);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&output_state_info, output_state_in);

    // Validate internal functions
    // _concat_input_weights
    std::vector<const ITensorInfo *> inputs_weights_vector;
    inputs_weights_vector.emplace_back(input_to_input_weights);
    inputs_weights_vector.emplace_back(input_to_forget_weights);
    inputs_weights_vector.emplace_back(input_to_cell_weights);
    inputs_weights_vector.emplace_back(input_to_output_weights);
    const QuantizationInfo qweights = input_to_input_weights->quantization_info(); // Weights quantization
    const TensorInfo       input_weights(TensorShape(input_size, 4 * output_size), 1, DataType::QASYMM8, qweights);
    ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(inputs_weights_vector, &input_weights, Window::DimY));

    // _concat_recurrent_weights
    std::vector<const ITensorInfo *> recurrent_weights_vector;
    recurrent_weights_vector.emplace_back(recurrent_to_input_weights);
    recurrent_weights_vector.emplace_back(recurrent_to_forget_weights);
    recurrent_weights_vector.emplace_back(recurrent_to_cell_weights);
    recurrent_weights_vector.emplace_back(recurrent_to_output_weights);
    const TensorInfo recurrent_weights(TensorShape(output_size, 4 * output_size), 1, DataType::QASYMM8, qweights);
    ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(recurrent_weights_vector, &recurrent_weights, Window::DimY));

    // _concat_weights
    std::vector<const ITensorInfo *> weights_vector;
    weights_vector.emplace_back(&recurrent_weights);
    weights_vector.emplace_back(&input_weights);
    const TensorInfo weights(TensorShape(input_size + output_size, 4 * output_size), 1, DataType::QASYMM8, qweights);
    ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(weights_vector, &weights, Window::DimX));
    // _transpose_weights
    const TensorShape weights_transposed_shape(weights.tensor_shape()[1], weights.tensor_shape()[0]);
    TensorInfo        weights_transposed = weights.clone()->set_is_resizable(true).set_tensor_shape(weights_transposed_shape);
    ARM_COMPUTE_RETURN_ON_ERROR(CLTranspose::validate(&weights, &weights_transposed));

    // _concat_inputs
    std::vector<const ITensorInfo *> input_vector;
    input_vector.emplace_back(input);
    input_vector.emplace_back(output_state_in);
    TensorInfo input_concatenated(TensorShape(output_size + input_size, batch_size), 1, DataType::QASYMM8, qasymm);
    ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(input_vector, &input_concatenated, Window::DimX));

    // _concat_bias
    std::vector<const ITensorInfo *> bias_vector;
    bias_vector.emplace_back(input_gate_bias);
    bias_vector.emplace_back(forget_gate_bias);
    bias_vector.emplace_back(cell_bias);
    bias_vector.emplace_back(output_gate_bias);

    const TensorInfo bias_concatenated(TensorShape(4 * output_size), 1, DataType::S32);
    ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(bias_vector, &bias_concatenated, Window::DimX));

    // Invert the offset for gemmlowp
    input_concatenated.set_quantization_info(QuantizationInfo(qasymm.uniform().scale, -qasymm.uniform().offset));
    weights_transposed.set_quantization_info(QuantizationInfo(qweights.uniform().scale, -qweights.uniform().offset));

    // _gemmlowp
    const TensorInfo output_highp(TensorShape(4 * output_size, batch_size), 1, DataType::S32);
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyCore::validate(&input_concatenated, &weights_transposed, nullptr, &output_highp));

    // Set the offset back
    input_concatenated.set_quantization_info(QuantizationInfo(qasymm.uniform().scale, qasymm.uniform().offset));
    weights_transposed.set_quantization_info(QuantizationInfo(qweights.uniform().scale, qweights.uniform().offset));

    // multiplier = (input_scale * weights_scale) / output_scale (2 ^ (-12))
    const TensorInfo output_lowp(output_highp.tensor_shape(), 1, DataType::QSYMM16, qsymm_3);

    const float multiplier = 4096.f * qasymm.uniform().scale * qweights.uniform().scale;
    ARM_COMPUTE_UNUSED(multiplier);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier > 1.0f);
    // _output_stage
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpQuantizeDownInt32ToInt16ScaleByFixedPoint::validate(&output_highp, &bias_concatenated, &output_lowp));

    TensorInfo input_gate_input;
    TensorInfo forget_gate_input;
    TensorInfo input_modulation_gate_input;
    TensorInfo output_gate_input;

    if(batch_size > 1)
    {
        // _slice_input_tensor
        input_gate_input = TensorInfo(TensorShape(output_size, batch_size), 1, DataType::QSYMM16, qsymm_3);
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&output_lowp, &input_gate_input, { 0, 0 }, { output_size, batch_size }));
        // _slice_forget_tensor
        forget_gate_input = TensorInfo(TensorShape(output_size, batch_size), 1, DataType::QSYMM16, qsymm_3);
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&output_lowp, &forget_gate_input, { output_size, 0 }, { 2 * output_size, batch_size }));
        // _slice_cell_tensor
        input_modulation_gate_input = TensorInfo(TensorShape(output_size, batch_size), 1, DataType::QSYMM16, qsymm_3);
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&output_lowp, &input_modulation_gate_input, { 2 * output_size, 0 }, { 3 * output_size, batch_size }));
        // _slice_output_tensor
        output_gate_input = TensorInfo(TensorShape(output_size, batch_size), 1, DataType::QSYMM16, qsymm_3);
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&output_lowp, &output_gate_input, { 3 * output_size, 0 }, { 4 * output_size, batch_size }));
    }
    else
    {
        // _slice_input_tensor
        input_gate_input = TensorInfo(TensorShape(output_size), 1, DataType::QSYMM16, qsymm_3);
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&output_lowp, &input_gate_input, { 0 }, { output_size }));
        // _slice_forget_tensor
        forget_gate_input = TensorInfo(TensorShape(output_size), 1, DataType::QSYMM16, qsymm_3);
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&output_lowp, &forget_gate_input, { output_size }, { 2 * output_size }));
        // _slice_cell_tensor
        input_modulation_gate_input = TensorInfo(TensorShape(output_size), 1, DataType::QSYMM16, qsymm_3);
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&output_lowp, &input_modulation_gate_input, { 2 * output_size }, { 3 * output_size }));
        // _slice_output_tensor
        output_gate_input = TensorInfo(TensorShape(output_size), 1, DataType::QSYMM16, qsymm_3);
        ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(&output_lowp, &output_gate_input, { 3 * output_size }, { 4 * output_size }));
    }

    // _sigmoid_forget_gate
    const TensorInfo forget_gate_output(forget_gate_input.tensor_shape(), 1, DataType::QSYMM16, qsymm_0);
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&forget_gate_input, &forget_gate_output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));
    // _sigmoid_input_gate
    const TensorInfo input_gate_output(input_gate_input.tensor_shape(), 1, DataType::QSYMM16, qsymm_0);
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&input_gate_input, &input_gate_output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));
    // _tanh_modulation_gate
    const TensorInfo input_modulation_gate_output(input_modulation_gate_input.tensor_shape(), 1, DataType::QSYMM16, qsymm_0);
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&input_modulation_gate_input, &input_modulation_gate_output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f)));
    // _sigmoid_output_gate
    const TensorInfo output_gate_output(output_gate_input.tensor_shape(), 1, DataType::QSYMM16, qsymm_0);
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&output_gate_input, &output_gate_output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));

    // _mul_forget_gate_cell_state
    const TensorInfo cell_state_tmp1(forget_gate_output.tensor_shape(), 1, DataType::QSYMM16, qsymm_4);
    ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&forget_gate_output, cell_state_in, &cell_state_tmp1, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO));

    // _mul_input_gate_input_mod_gate
    const TensorInfo cell_state_tmp2(input_gate_output.tensor_shape(), 1, DataType::QSYMM16, qsymm_4);
    ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&input_gate_output, &input_modulation_gate_output, &cell_state_tmp2, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO));

    // _add_cell_state_tmps
    ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&cell_state_tmp1, &cell_state_tmp2, cell_state_out, ConvertPolicy::SATURATE));

    // _tanh_modulation_gate
    const TensorInfo output_state_tmp(cell_state_out->tensor_shape(), 1, DataType::QSYMM16, qsymm_0);
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(cell_state_out, &output_state_tmp, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f)));

    // _mul_output_state_tmp_output_gate
    const TensorInfo output_state_out_symm(output_gate_output.tensor_shape(), 1, DataType::QSYMM16, qsymm_0);
    ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&output_state_tmp, &output_gate_output, &output_state_out_symm, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO));

    // _dequantize
    const TensorInfo output_state_out_f32(output_state_out_symm.tensor_shape(), 1, DataType::F32);
    ARM_COMPUTE_RETURN_ON_ERROR(CLDequantizationLayer::validate(&output_state_out_symm, &output_state_out_f32));

    // _quantize
    ARM_COMPUTE_RETURN_ON_ERROR(CLQuantizationLayer::validate(&output_state_out_f32, output_state_out));

    if(cell_state_out->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&cell_state_info, cell_state_out);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&cell_state_info, cell_state_out);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&cell_state_info, cell_state_out);
    }

    if(output_state_out->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&output_state_info, output_state_out);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&output_state_info, output_state_out);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&output_state_info, output_state_out);
    }

    return Status{};
}

void CLLSTMLayerQuantized::run()
{
    prepare();

    // Acquire all the temporaries
    MemoryGroupResourceScope scope_mg(_memory_group);

    // Concat and transpose the input
    _concat_inputs.run();

    // Run gemmlowp
    _gemmlowp.run();
    _output_stage.run();

    // Slice the results
    _slice_input_tensor.run();
    _slice_forget_tensor.run();
    _slice_cell_tensor.run();
    _slice_output_tensor.run();

    // Gates
    // Forget gate
    _sigmoid_forget_gate.run();

    // Input gate
    _sigmoid_input_gate.run();

    // Input modulation gate
    _tanh_modulation_gate.run();

    // Output gate
    _sigmoid_output_gate.run();

    // Cell state (long term memory)
    _mul_forget_gate_cell_state.run();
    _mul_input_gate_input_mod_gate.run();
    _add_cell_state_tmps.run();

    // Output state (short term memory)
    _tanh_output_state.run();
    _mul_output_state_tmp_output_gate.run();

    // Requantize output state from QSYMM16 to QASYMM16
    _dequantize.run();
    _quantize.run();
}

void CLLSTMLayerQuantized::prepare()
{
    if(!_is_prepared)
    {
        _input_weights.allocator()->allocate();
        _concat_input_weights.run();

        _input_to_input_weights->mark_as_unused();
        _input_to_forget_weights->mark_as_unused();
        _input_to_cell_weights->mark_as_unused();
        _input_to_output_weights->mark_as_unused();

        _recurrent_weights.allocator()->allocate();
        _concat_recurrent_weights.run();
        _recurrent_to_input_weights->mark_as_unused();
        _recurrent_to_forget_weights->mark_as_unused();
        _recurrent_to_cell_weights->mark_as_unused();
        _recurrent_to_output_weights->mark_as_unused();

        _weights.allocator()->allocate();
        _concat_weights.run();

        _input_weights.mark_as_unused();
        _input_weights.allocator()->free();
        _recurrent_weights.mark_as_unused();
        _recurrent_weights.allocator()->free();

        _weights_transposed.allocator()->allocate();
        _transpose_weights.run();

        _weights.mark_as_unused();
        _weights.allocator()->free();

        _bias.allocator()->allocate();
        _concat_bias.run();
        _input_gate_bias->mark_as_unused();
        _forget_gate_bias->mark_as_unused();
        _cell_bias->mark_as_unused();
        _output_gate_bias->mark_as_unused();

        _is_prepared = true;
    }
}

} // namespace arm_compute