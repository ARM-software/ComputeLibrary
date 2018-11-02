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
#include "arm_compute/runtime/CL/functions/CLLSTMLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLLSTMLayer::CLLSTMLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _fully_connected_input_gate(), _gemm_input_gate1(), _gemm_input_gate2(), _transpose_input_gate1(), _transpose_input_gate2(), _accum_input_gate1(),
      _accum_input_gate2(), _subtract_input_gate(), _activation_input_gate(), _fully_connected_forget_gate(), _gemm_forget_gate1(), _gemm_forget_gate2(), _transpose_forget_gate1(),
      _transpose_forget_gate2(), _accum_forget_gate1(), _accum_forget_gate2(), _activation_forget_gate(), _fully_connected_cell_state(), _gemm_cell_state1(), _gemm_cell_state2(), _transpose_cell_state1(),
      _accum_cell_state1(), _accum_cell_state2(), _pixelwise_mul_cell_state1(), _activation_cell_state(), _cell_clip(), _pixelwise_mul_cell_state2(), _fully_connected_output(), _gemm_output1(),
      _gemm_output2(), _transpose_output1(), _transpose_output2(), _accum_output1(), _accum_output2(), _activation_output(), _activation_output_state(), _pixelwise_mul_output_state(),
      _fully_connected_output_state(), _gemm_output_state(), _accum_output_state(), _projection_clip(), _copy_cell_state(), _copy_output(), _concat_scratch_buffer(), _input_gate_out1(), _input_gate_out2(),
      _input_gate_out3(), _input_gate_out4(), _input_gate_out5(), _input_gate_out6(), _forget_gate_out1(), _forget_gate_out2(), _forget_gate_out3(), _forget_gate_out4(), _forget_gate_out5(),
      _forget_gate_out6(), _cell_state_out1(), _cell_state_out2(), _cell_state_out3(), _cell_state_out4(), _cell_state_out5(), _output1(), _output2(), _output3(), _output4(), _output5(), _output6(),
      _cell_state_activation(), _output_projection1(), _ones(), _run_peephole_opt(false), _run_cifg_opt(false), _perform_cell_clipping(false), _has_projection_weights(false),
      _perform_projection_clipping(false)
{
}

void CLLSTMLayer::configure(const ICLTensor *input, const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                            const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                            const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                            ICLTensor *output_state, ICLTensor *cell_state, ICLTensor *scratch_buffer, ICLTensor *output, const LSTMParams<ICLTensor> &lstm_params, const ActivationLayerInfo &activation_info,
                            float cell_threshold, float projection_threshold)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights,
                                 forget_gate_bias, cell_bias, output_gate_bias, output_state, cell_state);
    LSTMParams<ITensorInfo> lstm_params_info;
    if(lstm_params.has_peephole_opt())
    {
        lstm_params_info.set_peephole_params(lstm_params.cell_to_input_weights()->info(), lstm_params.cell_to_forget_weights()->info(), lstm_params.cell_to_output_weights()->info());
    }
    if(lstm_params.has_projection())
    {
        lstm_params_info.set_projection_params(lstm_params.projection_weights()->info(), lstm_params.projection_bias()->info());
    }
    if(!lstm_params.has_cifg_opt())
    {
        lstm_params_info.set_cifg_params(lstm_params.input_to_input_weights()->info(), lstm_params.recurrent_to_input_weights()->info(),
                                         lstm_params.cell_to_input_weights()->info(), lstm_params.input_gate_bias()->info());
    }
    ARM_COMPUTE_ERROR_THROW_ON(CLLSTMLayer::validate(input->info(), input_to_forget_weights->info(),
                                                     input_to_cell_weights->info(), input_to_output_weights->info(),
                                                     recurrent_to_forget_weights->info(), recurrent_to_cell_weights->info(), recurrent_to_output_weights->info(),
                                                     forget_gate_bias->info(), cell_bias->info(), output_gate_bias->info(),
                                                     output_state->info(), cell_state->info(), scratch_buffer->info(), output->info(), lstm_params_info,
                                                     activation_info, cell_threshold, projection_threshold));

    const TensorShape cell_state_shape = cell_state->info()->tensor_shape();

    TensorShape forget_gate1_shape = compute_transposed_shape(*recurrent_to_output_weights->info());
    TensorShape forget_gate2_shape = compute_transposed_shape(*forget_gate_bias->info());
    TensorShape forget_gate3_shape{ 1, output_state->info()->dimension(1) };
    _forget_gate_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _forget_gate_out2.allocator()->init(TensorInfo(forget_gate1_shape, 1, input->info()->data_type()));
    _forget_gate_out3.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _forget_gate_out6.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

    // Configure block that calculates the forget gate
    // forget_gate = Activation(input * input_to_forget_weights + output_state * recurrent_to_forget_weights + cell_state * cell_to_forget_weights + forget_gate_bias)
    _memory_group.manage(&_forget_gate_out1);
    _fully_connected_forget_gate.configure(input, input_to_forget_weights, forget_gate_bias, &_forget_gate_out1, true, false);
    _memory_group.manage(&_forget_gate_out2);
    _transpose_forget_gate1.configure(recurrent_to_forget_weights, &_forget_gate_out2);
    _memory_group.manage(&_forget_gate_out3);
    _gemm_forget_gate1.configure(output_state, &_forget_gate_out2, nullptr, &_forget_gate_out3, 1.f, 0.f);
    _forget_gate_out2.allocator()->allocate();
    _memory_group.manage(&_forget_gate_out6);
    _accum_forget_gate1.configure(&_forget_gate_out1, &_forget_gate_out3, &_forget_gate_out6, ConvertPolicy::SATURATE);
    CLTensor *forget_gate_out = &_forget_gate_out6;

    if(lstm_params.has_peephole_opt())
    {
        _forget_gate_out4.allocator()->init(TensorInfo(forget_gate2_shape, 1, input->info()->data_type()));
        _forget_gate_out5.allocator()->init(TensorInfo(forget_gate3_shape, 1, input->info()->data_type()));

        _run_peephole_opt = true;
        _memory_group.manage(&_forget_gate_out4);
        _transpose_forget_gate2.configure(lstm_params.cell_to_forget_weights(), &_forget_gate_out4);
        _memory_group.manage(&_forget_gate_out5);
        _gemm_forget_gate2.configure(cell_state, &_forget_gate_out4, nullptr, &_forget_gate_out5, 1.f, 0.f);
        _forget_gate_out4.allocator()->allocate();
        _accum_forget_gate2.configure(&_forget_gate_out6, &_forget_gate_out5, &_forget_gate_out3, ConvertPolicy::SATURATE);
        _forget_gate_out5.allocator()->allocate();
        _forget_gate_out6.allocator()->allocate();
        forget_gate_out = &_forget_gate_out3;
    }
    else
    {
        _forget_gate_out3.allocator()->allocate();
    }
    _activation_forget_gate.configure(forget_gate_out, &_forget_gate_out1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    forget_gate_out->allocator()->allocate();

    TensorShape input_gate3_shape{ 1, output_state->info()->dimension(1) };
    _input_gate_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _input_gate_out5.allocator()->init(TensorInfo(input_gate3_shape, 1, input->info()->data_type()));

    // Configure block that calculates the input gate
    // input_gate = Activation(input * input_to_input_weights + output_state * recurrent_to_input_weights + cell_state * cell_to_input_weights + input_gate_bias), without CIFG
    // input_gate = 1 - forget_gate, with CIFG
    if(lstm_params.has_cifg_opt())
    {
        _memory_group.manage(&_input_gate_out1);
        _ones.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _subtract_input_gate.configure(&_ones, &_forget_gate_out1, &_input_gate_out1, ConvertPolicy::SATURATE);
        _ones.allocator()->allocate();
        _run_cifg_opt = true;
    }
    else
    {
        TensorShape input_gate1_shape = compute_transposed_shape(*recurrent_to_output_weights->info());
        TensorShape input_gate2_shape = compute_transposed_shape(*lstm_params.cell_to_input_weights()->info());

        _input_gate_out2.allocator()->init(TensorInfo(input_gate1_shape, 1, input->info()->data_type()));
        _input_gate_out3.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _input_gate_out4.allocator()->init(TensorInfo(input_gate2_shape, 1, input->info()->data_type()));
        _input_gate_out6.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

        _memory_group.manage(&_input_gate_out1);
        _fully_connected_input_gate.configure(input, lstm_params.input_to_input_weights(), lstm_params.input_gate_bias(), &_input_gate_out1, true, false);
        _memory_group.manage(&_input_gate_out2);
        _transpose_input_gate1.configure(lstm_params.recurrent_to_input_weights(), &_input_gate_out2);
        _memory_group.manage(&_input_gate_out3);
        _gemm_input_gate1.configure(output_state, &_input_gate_out2, nullptr, &_input_gate_out3, 1.f, 0.f);
        _input_gate_out2.allocator()->allocate();
        _memory_group.manage(&_input_gate_out4);
        _transpose_input_gate2.configure(lstm_params.cell_to_input_weights(), &_input_gate_out4);
        _memory_group.manage(&_input_gate_out5);
        _gemm_input_gate2.configure(cell_state, &_input_gate_out4, nullptr, &_input_gate_out5, 1.f, 0.f);
        _input_gate_out4.allocator()->allocate();
        _memory_group.manage(&_input_gate_out6);
        _accum_input_gate1.configure(&_input_gate_out1, &_input_gate_out3, &_input_gate_out6, ConvertPolicy::SATURATE);
        _input_gate_out3.allocator()->allocate();
        _accum_input_gate2.configure(&_input_gate_out6, &_input_gate_out5, &_input_gate_out1, ConvertPolicy::SATURATE);
        _input_gate_out5.allocator()->allocate();
        _input_gate_out6.allocator()->allocate();
        _activation_input_gate.configure(&_input_gate_out1, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    }

    TensorShape cell_state1_shape = compute_transposed_shape(*recurrent_to_output_weights->info());
    _cell_state_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _cell_state_out2.allocator()->init(TensorInfo(cell_state1_shape, 1, input->info()->data_type()));
    _cell_state_out3.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _cell_state_out4.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _cell_state_out5.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

    // Configure block that calculates the cell state
    // cell_state = Clip((RixelwiseMul(input_gate, Activation(input * input_to_cell_weights + output_state * recurrent_to_cell_weights + cell_bias)) + PixelwiseMul(forget_gate, cell_state)), cell_threshold)
    _memory_group.manage(&_cell_state_out1);
    _fully_connected_cell_state.configure(input, input_to_cell_weights, cell_bias, &_cell_state_out1, true, false);
    _memory_group.manage(&_cell_state_out2);
    _transpose_cell_state1.configure(recurrent_to_cell_weights, &_cell_state_out2);
    _memory_group.manage(&_cell_state_out3);
    _gemm_cell_state1.configure(output_state, &_cell_state_out2, nullptr, &_cell_state_out3, 1.f, 0.f);
    _cell_state_out2.allocator()->allocate();
    _memory_group.manage(&_cell_state_out4);
    _accum_cell_state1.configure(&_cell_state_out1, &_cell_state_out3, &_cell_state_out4, ConvertPolicy::SATURATE);
    _activation_cell_state.configure(&_cell_state_out4, nullptr, activation_info);
    _memory_group.manage(&_cell_state_out5);
    _pixelwise_mul_cell_state1.configure(&_cell_state_out4, &_input_gate_out1, &_cell_state_out5, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
    _input_gate_out1.allocator()->allocate();
    _cell_state_out4.allocator()->allocate();
    _pixelwise_mul_cell_state2.configure(&_forget_gate_out1, cell_state, &_cell_state_out3, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
    _forget_gate_out1.allocator()->allocate();
    _accum_cell_state2.configure(&_cell_state_out5, &_cell_state_out3, &_cell_state_out1, ConvertPolicy::SATURATE);
    _cell_state_out3.allocator()->allocate();
    _cell_state_out5.allocator()->allocate();

    // Perform clipping
    if(cell_threshold != 0.f)
    {
        _perform_cell_clipping = true;
        _cell_clip.configure(&_cell_state_out1, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -cell_threshold, cell_threshold));
    }

    TensorShape output1_shape = compute_transposed_shape(*recurrent_to_output_weights->info());
    TensorShape output2_shape = compute_transposed_shape(*cell_bias->info());
    TensorShape output3_shape{ 1, output_state->info()->dimension(1) };
    _output1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _output2.allocator()->init(TensorInfo(output1_shape, 1, input->info()->data_type()));
    _output3.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _output6.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

    // Configure block that calculates the output
    // output_gate = Activation(input * input_to_output_weights + output_state * recurrent_to_output_weights + cell_state * cell_to_output_weights + output_gate_bias)
    _memory_group.manage(&_output1);
    _fully_connected_output.configure(input, input_to_output_weights, output_gate_bias, &_output1, true, false);
    _memory_group.manage(&_output2);
    _transpose_output1.configure(recurrent_to_output_weights, &_output2);
    _memory_group.manage(&_output3);
    _gemm_output1.configure(output_state, &_output2, nullptr, &_output3, 1.f, 0.f);
    _output2.allocator()->allocate();
    _memory_group.manage(&_output6);
    _accum_output1.configure(&_output1, &_output3, &_output6, ConvertPolicy::SATURATE);
    _output3.allocator()->allocate();
    CLTensor *output_gate_out = &_output6;
    if(lstm_params.has_peephole_opt())
    {
        _output4.allocator()->init(TensorInfo(output2_shape, 1, input->info()->data_type()));
        _output5.allocator()->init(TensorInfo(output3_shape, 1, input->info()->data_type()));

        _memory_group.manage(&_output4);
        _transpose_output2.configure(lstm_params.cell_to_output_weights(), &_output4);
        _memory_group.manage(&_output5);
        _gemm_output2.configure(&_cell_state_out1, &_output4, nullptr, &_output5, 1.f, 0.f);
        _accum_output2.configure(&_output6, &_output5, &_output1, ConvertPolicy::SATURATE);
        _output6.allocator()->allocate();
        output_gate_out = &_output1;

        // Allocate intermediate buffers
        _output4.allocator()->allocate();
        _output5.allocator()->allocate();
    }
    else
    {
        _output1.allocator()->allocate();
    }
    _activation_output.configure(output_gate_out, output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    output_gate_out->allocator()->allocate();

    _cell_state_activation.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

    // Configure block that calculates the output state
    /** lstm_res = PixelwiseMul(output, Activation(cell_state))
     *
     *                      -- Clip(lstm_res * projection_weights + projection_bias, projection_threshold) , if there is a projection
     *                     /
     *  output_state =  --
     *                     \
     *                      -- lstm_res , otherwise
     */
    _memory_group.manage(&_cell_state_activation);
    _activation_output_state.configure(&_cell_state_out1, &_cell_state_activation, activation_info);
    _pixelwise_mul_output_state.configure(&_cell_state_activation, output, output_state, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
    _cell_state_activation.allocator()->allocate();

    if(lstm_params.has_projection())
    {
        _has_projection_weights = true;
        _output_projection1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _memory_group.manage(&_output_projection1);
        _fully_connected_output_state.configure(output_state, lstm_params.projection_weights(), lstm_params.projection_bias(), &_output_projection1, true, false);
        // Perform clipping
        if(projection_threshold != 0.f)
        {
            _perform_projection_clipping = true;
            _projection_clip.configure(&_output_projection1, output_state, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -projection_threshold, projection_threshold));
        }

        // Allocate intermediate buffer
        _output_projection1.allocator()->allocate();
    }

    // Copy cell state and output
    _copy_cell_state.configure(&_cell_state_out1, cell_state);
    _cell_state_out1.allocator()->allocate();
    _copy_output.configure(output_state, output);

    // Vector for holding the tensors to store in scratch buffer
    std::vector<ICLTensor *> scratch_inputs;
    if(lstm_params.has_cifg_opt())
    {
        scratch_inputs.emplace_back(&_input_gate_out1);
    }
    scratch_inputs.emplace_back(&_cell_state_out1);
    scratch_inputs.emplace_back(forget_gate_out);
    scratch_inputs.emplace_back(output_gate_out);
    _concat_scratch_buffer.configure(scratch_inputs, scratch_buffer);
}

Status CLLSTMLayer::validate(const ITensorInfo *input, const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                             const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                             const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                             const ITensorInfo *output_state, const ITensorInfo *cell_state, const ITensorInfo *scratch_buffer, const ITensorInfo *output,
                             const LSTMParams<ITensorInfo> &lstm_params, const ActivationLayerInfo &activation_info, float cell_threshold, float projection_threshold)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights,
                                        forget_gate_bias, cell_bias, output_gate_bias, output_state, cell_state);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, recurrent_to_forget_weights, recurrent_to_cell_weights,
                                                       recurrent_to_output_weights, forget_gate_bias, cell_bias, output_gate_bias, output_state, cell_state);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_to_forget_weights->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_to_cell_weights->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_to_output_weights->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_to_forget_weights->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_to_cell_weights->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_to_output_weights->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(forget_gate_bias->num_dimensions() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_bias->num_dimensions() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(output_gate_bias->num_dimensions() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(output_state->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_state->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(scratch_buffer->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_bias->dimension(0) * 4 != scratch_buffer->dimension(0) && cell_bias->dimension(0) * 3 != scratch_buffer->dimension(0));

    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.cell_to_input_weights(), lstm_params.cell_to_output_weights(), lstm_params.cell_to_forget_weights());
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_input_weights()->num_dimensions() != 1);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_forget_weights()->num_dimensions() != 1);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_output_weights()->num_dimensions() != 1);
    }

    TensorShape      units_out_transposed_shape = compute_transposed_shape(*recurrent_to_output_weights);
    TensorShape      gemmv_shape{ 1, output_state->dimension(1) };
    TensorShape      num_units_transposed_shape = compute_transposed_shape(*forget_gate_bias);
    const TensorInfo units_out_transposed_info  = TensorInfo(units_out_transposed_shape, 1, input->data_type());
    const TensorInfo gemmv_shape_info           = TensorInfo(gemmv_shape, 1, input->data_type());
    const TensorInfo num_units_transposed_info  = TensorInfo(num_units_transposed_shape, 1, input->data_type());

    // Validate forget gate
    ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, input_to_forget_weights, forget_gate_bias, cell_state, true, false));
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(output_state, &units_out_transposed_info, nullptr, cell_state, 1.f, 0.f, GEMMInfo()));
    ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAdditionKernel::validate(cell_state, cell_state, cell_state, ConvertPolicy::SATURATE));
    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(cell_state, &num_units_transposed_info, nullptr, &gemmv_shape_info, 1.f, 0.f, GEMMInfo()));
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(cell_state, &gemmv_shape_info, cell_state, ConvertPolicy::SATURATE));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayerKernel::validate(cell_state, cell_state, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));

    // Validate input gate
    if(!lstm_params.has_cifg_opt())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.input_to_input_weights(), lstm_params.recurrent_to_input_weights(), lstm_params.cell_to_input_weights(), lstm_params.input_gate_bias());
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.input_to_input_weights()->num_dimensions() != 2);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.recurrent_to_input_weights()->num_dimensions() != 2);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_input_weights()->num_dimensions() != 1);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.input_gate_bias()->num_dimensions() != 1);
        ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, lstm_params.input_to_input_weights(), lstm_params.input_gate_bias(), cell_state, true, false));
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(cell_state, &num_units_transposed_info, nullptr, &gemmv_shape_info, 1.f, 0.f, GEMMInfo()));
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(cell_state, &gemmv_shape_info, cell_state, ConvertPolicy::SATURATE));
        ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayerKernel::validate(cell_state, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticSubtractionKernel::validate(cell_state, cell_state, cell_state, ConvertPolicy::SATURATE));
    }

    // Validate cell state
    ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, input_to_cell_weights, cell_bias, cell_state, true, false));
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayerKernel::validate(cell_state, nullptr, activation_info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplicationKernel::validate(cell_state, cell_state, cell_state, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN));

    if(cell_threshold != 0.f)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayerKernel::validate(cell_state, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -cell_threshold, cell_threshold)));
    }

    ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, input_to_output_weights, output_gate_bias, cell_state, true, false));
    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(cell_state, cell_state, cell_state, ConvertPolicy::SATURATE));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayerKernel::validate(cell_state, output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));

    // Validate output state
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayerKernel::validate(cell_state, cell_state, activation_info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplicationKernel::validate(cell_state, output, output_state, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN));
    if(lstm_params.has_projection())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(output_state, lstm_params.projection_weights(), lstm_params.projection_bias(), cell_state, true, false));
        if(projection_threshold != 0.f)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayerKernel::validate(cell_state, output_state, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -projection_threshold,
                                                                                                                        projection_threshold)));
        }
    }

    std::vector<TensorInfo> inputs_vector_info;
    if(lstm_params.has_cifg_opt())
    {
        inputs_vector_info.emplace_back(*cell_state);
    }
    inputs_vector_info.emplace_back(*cell_state);
    inputs_vector_info.emplace_back(*cell_state);
    inputs_vector_info.emplace_back(*cell_state);

    std::vector<ITensorInfo *> inputs_vector_info_raw;
    for(auto &input : inputs_vector_info)
    {
        inputs_vector_info_raw.emplace_back(&input);
    }

    ARM_COMPUTE_RETURN_ON_ERROR(CLWidthConcatenateLayer::validate(inputs_vector_info_raw, scratch_buffer));
    return Status{};
}

void CLLSTMLayer::run()
{
    _memory_group.acquire();

    _fully_connected_forget_gate.run();
    CLScheduler::get().enqueue(_transpose_forget_gate1);
    _gemm_forget_gate1.run();
    CLScheduler::get().enqueue(_accum_forget_gate1);

    if(_run_peephole_opt)
    {
        CLScheduler::get().enqueue(_transpose_forget_gate2);
        _gemm_forget_gate2.run();
        _accum_forget_gate2.run();
    }
    CLScheduler::get().enqueue(_activation_forget_gate);

    if(_run_cifg_opt)
    {
        _ones.map(true);
        std::fill_n(_ones.buffer(), _ones.info()->total_size(), 1);
        _ones.unmap();
        CLScheduler::get().enqueue(_subtract_input_gate);
    }
    else
    {
        _fully_connected_input_gate.run();
        CLScheduler::get().enqueue(_transpose_input_gate1);
        _gemm_input_gate1.run();
        CLScheduler::get().enqueue(_transpose_input_gate2);
        _gemm_input_gate2.run();
        CLScheduler::get().enqueue(_accum_input_gate1);
        _accum_input_gate2.run();
        CLScheduler::get().enqueue(_activation_input_gate);
    }

    _fully_connected_cell_state.run();
    CLScheduler::get().enqueue(_transpose_cell_state1);
    _gemm_cell_state1.run();
    CLScheduler::get().enqueue(_accum_cell_state1);
    CLScheduler::get().enqueue(_activation_cell_state);
    CLScheduler::get().enqueue(_pixelwise_mul_cell_state1);
    CLScheduler::get().enqueue(_pixelwise_mul_cell_state2);
    CLScheduler::get().enqueue(_accum_cell_state2);

    if(_perform_cell_clipping)
    {
        CLScheduler::get().enqueue(_cell_clip);
    }

    _fully_connected_output.run();
    CLScheduler::get().enqueue(_transpose_output1);
    _gemm_output1.run();
    CLScheduler::get().enqueue(_accum_output1);
    CLScheduler::get().enqueue(_pixelwise_mul_output_state);

    if(_run_peephole_opt)
    {
        CLScheduler::get().enqueue(_transpose_output2);
        _gemm_output2.run();
        _accum_output2.run();
    }
    CLScheduler::get().enqueue(_activation_output);

    CLScheduler::get().enqueue(_activation_output_state);
    CLScheduler::get().enqueue(_pixelwise_mul_output_state);

    if(_has_projection_weights)
    {
        _fully_connected_output_state.run();
        if(_perform_projection_clipping)
        {
            CLScheduler::get().enqueue(_projection_clip);
        }
    }

    CLScheduler::get().enqueue(_copy_cell_state);
    CLScheduler::get().enqueue(_copy_output);

    _concat_scratch_buffer.run();

    _memory_group.release();
}