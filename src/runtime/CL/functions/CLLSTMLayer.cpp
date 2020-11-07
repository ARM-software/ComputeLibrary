/*
 * Copyright (c) 2018-2020 Arm Limited.
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

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/InfoHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLCopyKernel.h"
#include "src/core/CL/kernels/CLDepthConvertLayerKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpMatrixMultiplyNativeKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpOffsetContributionKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/CL/kernels/CLGEMMLowpReductionKernel.h"
#include "src/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "src/core/CL/kernels/CLGEMMMatrixMultiplyReshapedKernel.h"
#include "src/core/CL/kernels/CLGEMMMatrixMultiplyReshapedOnlyRHSKernel.h"
#include "src/core/CL/kernels/CLGEMMReshapeLHSMatrixKernel.h"
#include "src/core/CL/kernels/CLGEMMReshapeRHSMatrixKernel.h"
#include "src/core/CL/kernels/CLMemsetKernel.h"
#include "src/core/CL/kernels/CLTransposeKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::utils::info_helpers;

CLLSTMLayer::CLLSTMLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _fully_connected_input_gate(), _accum_input_gate1(), _subtract_input_gate(), _pixelwise_mul_input_gate(), _activation_input_gate(),
      _fully_connected_forget_gate(), _accum_forget_gate1(), _pixelwise_mul_forget_gate(), _activation_forget_gate(), _fully_connected_cell_state(), _gemm_cell_state1(),
      _transpose_cell_state(support::cpp14::make_unique<CLTransposeKernel>()), _accum_cell_state1(), _accum_cell_state2(), _pixelwise_mul_cell_state1(), _activation_cell_state(), _cell_clip(),
      _pixelwise_mul_cell_state2(), _fully_connected_output(), _pixelwise_mul_output_state1(), _accum_output1(), _activation_output(), _activation_output_state(), _pixelwise_mul_output_state2(),
      _fully_connected_output_state(), _projection_clip(), _copy_cell_state(support::cpp14::make_unique<CLCopyKernel>()), _copy_output(support::cpp14::make_unique<CLCopyKernel>()), _concat_scratch_buffer(),
      _concat_inputs_forget_gate(), _concat_weights_forget_gate(), _concat_weights_input_gate(), _concat_weights_output(), _ones_memset_kernel(support::cpp14::make_unique<CLMemsetKernel>()),
      _mean_std_norm_input_gate(), _pixelwise_mul_input_gate_coeff(), _accum_input_gate_bias(), _mean_std_norm_forget_gate(), _pixelwise_mul_forget_gate_coeff(), _accum_forget_gate_bias(),
      _mean_std_norm_cell_gate(), _pixelwise_mul_cell_gate_coeff(), _accum_cell_gate_bias(), _mean_std_norm_output_gate(), _pixelwise_mul_output_gate_coeff(), _accum_output_gate_bias(), _input_gate_out1(),
      _input_gate_out2(), _input_gate_out3(), _input_gate_out4(), _forget_gate_out1(), _forget_gate_out2(), _forget_gate_out3(), _forget_gate_out4(), _forget_gate_out5(), _forget_gate_out6(),
      _cell_state_out1(), _cell_state_out2(), _cell_state_out3(), _cell_state_out4(), _cell_state_out5(), _output1(), _output2(), _output3(), _output4(), _cell_state_activation(), _output_state1(), _ones(),
      _input_layer_norm_out1(), _input_layer_norm_out2(), _forget_layer_norm_out1(), _forget_layer_norm_out2(), _cell_layer_norm_out1(), _cell_layer_norm_out2(), _output_layer_norm_out1(),
      _output_layer_norm_out2(), _run_peephole_opt(false), _run_cifg_opt(false), _perform_cell_clipping(false), _has_projection_weights(false), _perform_projection_clipping(false), _is_prepared(false),
      _is_layer_norm_lstm(false)
{
}

CLLSTMLayer::~CLLSTMLayer() = default;

void CLLSTMLayer::configure(const ICLTensor *input,
                            const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                            const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                            const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                            const ICLTensor *output_state_in, ICLTensor *cell_state_in,
                            ICLTensor *scratch_buffer, ICLTensor *output_state_out, ICLTensor *cell_state_out, ICLTensor *output,
                            const LSTMParams<ICLTensor> &lstm_params, const ActivationLayerInfo &activation_info, float cell_threshold, float projection_threshold)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, recurrent_to_forget_weights, recurrent_to_cell_weights,
              recurrent_to_output_weights, forget_gate_bias, cell_bias, output_gate_bias, output_state_in, cell_state_in, scratch_buffer, output_state_out, cell_state_out, output, lstm_params, activation_info,
              cell_threshold, projection_threshold);
}

void CLLSTMLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input,
                            const ICLTensor *input_to_forget_weights, const ICLTensor *input_to_cell_weights, const ICLTensor *input_to_output_weights,
                            const ICLTensor *recurrent_to_forget_weights, const ICLTensor *recurrent_to_cell_weights, const ICLTensor *recurrent_to_output_weights,
                            const ICLTensor *forget_gate_bias, const ICLTensor *cell_bias, const ICLTensor *output_gate_bias,
                            const ICLTensor *output_state_in, ICLTensor *cell_state_in,
                            ICLTensor *scratch_buffer, ICLTensor *output_state_out, ICLTensor *cell_state_out, ICLTensor *output,
                            const LSTMParams<ICLTensor> &lstm_params, const ActivationLayerInfo &activation_info, float cell_threshold, float projection_threshold)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input,
                                 input_to_forget_weights, input_to_cell_weights, input_to_output_weights,
                                 recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights,
                                 forget_gate_bias, cell_bias, output_gate_bias,
                                 output_state_in, cell_state_in,
                                 scratch_buffer, output_state_out, cell_state_out, output);

    _is_layer_norm_lstm = lstm_params.use_layer_norm();

    // Set lstm parameters
    LSTMParams<ITensorInfo> lstm_params_info{};
    build_lstm_params_tensor_info(lstm_params, &lstm_params_info);

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(CLLSTMLayer::validate(input->info(), input_to_forget_weights->info(),
                                                     input_to_cell_weights->info(), input_to_output_weights->info(),
                                                     recurrent_to_forget_weights->info(), recurrent_to_cell_weights->info(), recurrent_to_output_weights->info(),
                                                     forget_gate_bias->info(), cell_bias->info(), output_gate_bias->info(),
                                                     output_state_in->info(), cell_state_in->info(),
                                                     scratch_buffer->info(), output_state_out->info(), cell_state_out->info(), output->info(),
                                                     lstm_params_info, activation_info, cell_threshold, projection_threshold));

    const TensorShape cell_state_shape = cell_state_in->info()->tensor_shape();
    // Configure block that calculates the forget gate
    // forget_gate = Activation(input * input_to_forget_weights + output_state_in * recurrent_to_forget_weights + PixelWiseMul(cell_state, cell_to_forget_weights) + forget_gate_bias)
    // We optimize this as follows:
    // forget_gate = Activation( (input,output_state_in) * (input_to_forget_weights,recurrent_to_forget_weights) + PixelWiseMul(cell_state, cell_to_forget_weights) + forget_gate_bias
    _forget_gate_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _forget_gate_out3.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _forget_gate_out5.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

    std::vector<const ICLTensor *> inputs_vector;
    inputs_vector.emplace_back(input);
    inputs_vector.emplace_back(output_state_in);
    const TensorShape concat_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, 0);
    _forget_gate_out2.allocator()->init(TensorInfo(concat_shape, 1, input->info()->data_type()));

    _memory_group.manage(&_forget_gate_out2);
    _concat_inputs_forget_gate.configure(compile_context, inputs_vector, &_forget_gate_out2, Window::DimX);

    std::vector<const ICLTensor *> weights_vector;

    weights_vector.emplace_back(input_to_forget_weights);
    weights_vector.emplace_back(recurrent_to_forget_weights);
    const TensorShape weights_concat_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(weights_vector, 0);
    _forget_gate_out6.allocator()->init(TensorInfo(weights_concat_shape, 1, input->info()->data_type()));

    _concat_weights_forget_gate.configure(compile_context, weights_vector, &_forget_gate_out6, Window::DimX);

    _memory_group.manage(&_forget_gate_out5);
    _fully_connected_forget_gate.configure(compile_context, &_forget_gate_out2, &_forget_gate_out6, (_is_layer_norm_lstm) ? nullptr : forget_gate_bias, &_forget_gate_out5);
    _memory_group.manage(&_forget_gate_out1);
    _memory_group.manage(&_forget_gate_out3);
    _forget_gate_out6.allocator()->allocate();

    CLTensor *forget_gate_out = &_forget_gate_out5;
    if(lstm_params.has_peephole_opt())
    {
        _forget_gate_out4.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

        _run_peephole_opt = true;
        _memory_group.manage(&_forget_gate_out4);
        _pixelwise_mul_forget_gate.configure(compile_context, cell_state_in, lstm_params.cell_to_forget_weights(), &_forget_gate_out4, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
        _accum_forget_gate1.configure(compile_context, &_forget_gate_out5, &_forget_gate_out4, &_forget_gate_out3, ConvertPolicy::SATURATE);
        _forget_gate_out4.allocator()->allocate();
        _forget_gate_out5.allocator()->allocate();
        forget_gate_out = &_forget_gate_out3;
    }
    else
    {
        _forget_gate_out3.allocator()->allocate();
    }
    if(_is_layer_norm_lstm)
    {
        _forget_layer_norm_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _forget_layer_norm_out2.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _memory_group.manage(&_forget_layer_norm_out1);
        _memory_group.manage(&_forget_layer_norm_out2);
        _mean_std_norm_forget_gate.configure(compile_context, forget_gate_out);
        _pixelwise_mul_forget_gate_coeff.configure(compile_context, forget_gate_out, lstm_params.forget_layer_norm_weights(), &_forget_layer_norm_out1, 1, ConvertPolicy::SATURATE,
                                                   RoundingPolicy::TO_NEAREST_EVEN);
        // forget_gate_out is going to be reassigned, so allocate the tensor that it was assigned to before
        forget_gate_out->allocator()->allocate();
        _accum_forget_gate_bias.configure(compile_context, &_forget_layer_norm_out1, forget_gate_bias, &_forget_layer_norm_out2, ConvertPolicy::SATURATE);
        _forget_layer_norm_out1.allocator()->allocate();
        forget_gate_out = &_forget_layer_norm_out2;
    }
    _activation_forget_gate.configure(compile_context, forget_gate_out, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));

    // Configure block that calculates the input gate
    // input_gate = Activation(input * input_to_input_weights + output_state * recurrent_to_input_weights + PixelWiseMul(cell_state, cell_to_input_weights) + input_gate_bias), without CIFG
    // input_gate = 1 - forget_gate, with CIFG
    // We optimize this as follows:
    // input_gate = Activation((input,output_state) * (input_to_input_weights,recurrent_to_input_weights) + PixelWiseMul(cell_state, cell_to_input_weights) + input_gate_bias), without CIFG
    _input_gate_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    CLTensor *input_gate_out = &_input_gate_out1;
    if(lstm_params.has_cifg_opt())
    {
        _memory_group.manage(&_input_gate_out1);
        _ones.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _ones_memset_kernel->configure(compile_context, &_ones, PixelValue(1, _ones.info()->data_type()));
        _subtract_input_gate.configure(compile_context, &_ones, forget_gate_out, &_input_gate_out1, ConvertPolicy::SATURATE);
        _ones.allocator()->allocate();
        _run_cifg_opt = true;
    }
    else
    {
        _input_gate_out3.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _input_gate_out4.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

        std::vector<const ICLTensor *> lstm_weights;
        lstm_weights.emplace_back(lstm_params.input_to_input_weights());
        lstm_weights.emplace_back(lstm_params.recurrent_to_input_weights());
        TensorShape lstm_weights_concat_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(lstm_weights, 0);
        _input_gate_out2.allocator()->init(TensorInfo(lstm_weights_concat_shape, 1, input->info()->data_type()));

        _concat_weights_input_gate.configure(compile_context, lstm_weights, &_input_gate_out2, Window::DimX);

        _memory_group.manage(&_input_gate_out1);

        _memory_group.manage(&_input_gate_out3);
        _fully_connected_input_gate.configure(compile_context, &_forget_gate_out2, &_input_gate_out2, (_is_layer_norm_lstm) ? nullptr : lstm_params.input_gate_bias(), &_input_gate_out3);
        _input_gate_out2.allocator()->allocate();

        input_gate_out = &_input_gate_out3;
        if(_run_peephole_opt)
        {
            _memory_group.manage(&_input_gate_out4);
            _pixelwise_mul_input_gate.configure(compile_context, cell_state_in, lstm_params.cell_to_input_weights(), &_input_gate_out4, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
            _accum_input_gate1.configure(compile_context, &_input_gate_out3, &_input_gate_out4, &_input_gate_out1, ConvertPolicy::SATURATE);
            _input_gate_out3.allocator()->allocate();
            _input_gate_out4.allocator()->allocate();
            input_gate_out = &_input_gate_out1;
        }
        else
        {
            _input_gate_out1.allocator()->allocate();
        }

        if(_is_layer_norm_lstm)
        {
            _input_layer_norm_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
            _input_layer_norm_out2.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
            _memory_group.manage(&_input_layer_norm_out1);
            _memory_group.manage(&_input_layer_norm_out2);
            _mean_std_norm_input_gate.configure(compile_context, input_gate_out);
            _pixelwise_mul_input_gate_coeff.configure(compile_context, input_gate_out, lstm_params.input_layer_norm_weights(), &_input_layer_norm_out1, 1, ConvertPolicy::SATURATE,
                                                      RoundingPolicy::TO_NEAREST_EVEN);
            // input_gate_out is going to be reassigned, so allocate the tensor that it was assigned to before
            input_gate_out->allocator()->allocate();
            _accum_input_gate_bias.configure(compile_context, &_input_layer_norm_out1, lstm_params.input_gate_bias(), &_input_layer_norm_out2, ConvertPolicy::SATURATE);
            _input_layer_norm_out1.allocator()->allocate();
            input_gate_out = &_input_layer_norm_out2;
        }
        _activation_input_gate.configure(compile_context, input_gate_out, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    }

    // Configure block that calculates the cell state
    // cell_state = Clip((PixelwiseMul(input_gate, Activation(input * input_to_cell_weights + output_state_in * recurrent_to_cell_weights + cell_bias)) + PixelwiseMul(forget_gate, cell_state)), cell_threshold)
    TensorShape cell_state1_shape = compute_transposed_shape(*recurrent_to_output_weights->info());
    _cell_state_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _cell_state_out2.allocator()->init(TensorInfo(cell_state1_shape, 1, input->info()->data_type()));
    _cell_state_out3.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _cell_state_out4.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _cell_state_out5.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

    _memory_group.manage(&_cell_state_out1);
    _fully_connected_cell_state.configure(compile_context, input, input_to_cell_weights, (_is_layer_norm_lstm) ? nullptr : cell_bias, &_cell_state_out1);
    _memory_group.manage(&_cell_state_out2);
    _transpose_cell_state->configure(compile_context, recurrent_to_cell_weights, &_cell_state_out2);
    _memory_group.manage(&_cell_state_out3);
    _gemm_cell_state1.configure(compile_context, output_state_in, &_cell_state_out2, nullptr, &_cell_state_out3, 1.f, 0.f);
    _cell_state_out2.allocator()->allocate();
    _memory_group.manage(&_cell_state_out4);
    _accum_cell_state1.configure(compile_context, &_cell_state_out1, &_cell_state_out3, &_cell_state_out4, ConvertPolicy::SATURATE);
    CLTensor *cell_state_out_ptr = &_cell_state_out4;
    if(_is_layer_norm_lstm)
    {
        _cell_layer_norm_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _cell_layer_norm_out2.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _memory_group.manage(&_cell_layer_norm_out1);
        _memory_group.manage(&_cell_layer_norm_out2);
        _mean_std_norm_cell_gate.configure(compile_context, cell_state_out_ptr);
        _pixelwise_mul_cell_gate_coeff.configure(compile_context, cell_state_out_ptr, lstm_params.cell_layer_norm_weights(), &_cell_layer_norm_out1, 1, ConvertPolicy::SATURATE,
                                                 RoundingPolicy::TO_NEAREST_EVEN);
        // cell_state_out_ptr is going to be reassigned, so allocate the tensor that it was assigned to before
        cell_state_out_ptr->allocator()->allocate();
        _accum_cell_gate_bias.configure(compile_context, &_cell_layer_norm_out1, cell_bias, &_cell_layer_norm_out2, ConvertPolicy::SATURATE);
        _cell_layer_norm_out1.allocator()->allocate();
        cell_state_out_ptr = &_cell_layer_norm_out2;
    }
    _activation_cell_state.configure(compile_context, cell_state_out_ptr, nullptr, activation_info);
    _memory_group.manage(&_cell_state_out5);
    _pixelwise_mul_cell_state1.configure(compile_context, cell_state_out_ptr, input_gate_out, &_cell_state_out5, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
    cell_state_out_ptr->allocator()->allocate();
    _pixelwise_mul_cell_state2.configure(compile_context, forget_gate_out, cell_state_in, &_cell_state_out3, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
    _accum_cell_state2.configure(compile_context, &_cell_state_out5, &_cell_state_out3, &_cell_state_out1, ConvertPolicy::SATURATE);
    _cell_state_out3.allocator()->allocate();
    _cell_state_out5.allocator()->allocate();
    // Perform clipping
    if(cell_threshold != 0.f)
    {
        _perform_cell_clipping = true;
        _cell_clip.configure(compile_context, &_cell_state_out1, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -cell_threshold, cell_threshold));
    }

    // Configure block that calculates the output
    // output_state_out = Activation(input * input_to_output_weights + output_state_in * recurrent_to_output_weights + PixelWiseMul(cell_state, cell_to_output_weights) + output_gate_bias)
    // We optimize this as follows:
    // output_state_out = Activation( (input,output_state_in) * (input_to_output_weights, recurrent_to_output_weights) + PixelWiseMul(cell_state, cell_to_output_weights) + output_gate_bias)
    _output1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _output4.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    std::vector<const ICLTensor *> in_out_weights;
    in_out_weights.emplace_back(input_to_output_weights);
    in_out_weights.emplace_back(recurrent_to_output_weights);
    TensorShape in_out_weights_concat_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(in_out_weights, 0);
    _output2.allocator()->init(TensorInfo(in_out_weights_concat_shape, 1, input->info()->data_type()));

    _concat_weights_output.configure(compile_context, in_out_weights, &_output2, Window::DimX);

    _memory_group.manage(&_output1);
    _memory_group.manage(&_output4);

    _fully_connected_output.configure(compile_context, &_forget_gate_out2, &_output2, (_is_layer_norm_lstm) ? nullptr : output_gate_bias, &_output4);

    _output2.allocator()->allocate();
    _forget_gate_out2.allocator()->allocate();

    CLTensor *output_gate_out = &_output4;
    if(lstm_params.has_peephole_opt())
    {
        _output3.allocator()->init(TensorInfo(_cell_state_out1.info()->tensor_shape(), 1, input->info()->data_type()));

        _memory_group.manage(&_output3);
        _pixelwise_mul_output_state1.configure(compile_context, &_cell_state_out1, lstm_params.cell_to_output_weights(), &_output3, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
        _accum_output1.configure(compile_context, &_output4, &_output3, &_output1, ConvertPolicy::SATURATE);
        _output4.allocator()->allocate();
        output_gate_out = &_output1;

        // Allocate intermediate buffers
        _output3.allocator()->allocate();
    }
    else
    {
        _output1.allocator()->allocate();
    }
    if(_is_layer_norm_lstm)
    {
        _output_layer_norm_out1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _output_layer_norm_out2.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
        _memory_group.manage(&_output_layer_norm_out1);
        _memory_group.manage(&_output_layer_norm_out2);
        _mean_std_norm_output_gate.configure(compile_context, output_gate_out);
        _pixelwise_mul_output_gate_coeff.configure(compile_context, output_gate_out, lstm_params.output_layer_norm_weights(), &_output_layer_norm_out1, 1, ConvertPolicy::SATURATE,
                                                   RoundingPolicy::TO_NEAREST_EVEN);
        // output_gate_out is going to be reassigned, so allocate the tensor that it was assigned to before
        output_gate_out->allocator()->allocate();
        _accum_output_gate_bias.configure(compile_context, &_output_layer_norm_out1, output_gate_bias, &_output_layer_norm_out2, ConvertPolicy::SATURATE);
        _output_layer_norm_out1.allocator()->allocate();
        output_gate_out = &_output_layer_norm_out2;
    }
    _activation_output.configure(compile_context, output_gate_out, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));

    // Configure block that calculates the output state
    /** lstm_res = PixelwiseMul(output, Activation(cell_state))
     *
     *                      -- Clip(lstm_res * projection_weights + projection_bias, projection_threshold) , if there is a projection
     *                     /
     *  output_state =  --
     *                     \
     *                      -- lstm_res , otherwise
     */
    ICLTensor *output_state_out_tmp = lstm_params.has_projection() ? &_output_state1 : output_state_out;
    _cell_state_activation.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));
    _output_state1.allocator()->init(TensorInfo(cell_state_shape, 1, input->info()->data_type()));

    _memory_group.manage(&_cell_state_activation);
    _activation_output_state.configure(compile_context, &_cell_state_out1, &_cell_state_activation, activation_info);
    _pixelwise_mul_output_state2.configure(compile_context, &_cell_state_activation, output_gate_out, output_state_out_tmp, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN);
    _cell_state_activation.allocator()->allocate();

    if(lstm_params.has_projection())
    {
        _has_projection_weights = true;
        _fully_connected_output_state.configure(compile_context, output_state_out_tmp, lstm_params.projection_weights(), lstm_params.projection_bias(), output_state_out);
        _output_state1.allocator()->allocate();
        // Perform clipping
        if(projection_threshold != 0.f)
        {
            _perform_projection_clipping = true;
            _projection_clip.configure(compile_context, output_state_out, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -projection_threshold, projection_threshold));
        }
    }

    // Copy cell state and output
    _copy_cell_state->configure(compile_context, &_cell_state_out1, cell_state_out);
    _copy_output->configure(compile_context, output_state_out, output);

    // Vector for holding the tensors to store in scratch buffer
    std::vector<const ICLTensor *> scratch_inputs;
    if(!lstm_params.has_cifg_opt())
    {
        scratch_inputs.emplace_back(input_gate_out);
    }
    scratch_inputs.emplace_back(&_cell_state_out1);
    scratch_inputs.emplace_back(forget_gate_out);
    scratch_inputs.emplace_back(output_gate_out);
    _concat_scratch_buffer.configure(compile_context, scratch_inputs, scratch_buffer, Window::DimX);
    input_gate_out->allocator()->allocate();
    _cell_state_out1.allocator()->allocate();
    forget_gate_out->allocator()->allocate();
    output_gate_out->allocator()->allocate();
}

Status CLLSTMLayer::validate(const ITensorInfo *input,
                             const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                             const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                             const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                             const ITensorInfo *output_state_in, const ITensorInfo *cell_state_in,
                             const ITensorInfo *scratch_buffer, const ITensorInfo *output_state_out, const ITensorInfo *cell_state_out, const ITensorInfo *output,
                             const LSTMParams<ITensorInfo> &lstm_params, const ActivationLayerInfo &activation_info, float cell_threshold, float projection_threshold)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input,
                                        input_to_forget_weights, input_to_cell_weights, input_to_output_weights,
                                        recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights,
                                        forget_gate_bias, cell_bias, output_gate_bias,
                                        output_state_in, cell_state_in,
                                        scratch_buffer, output_state_out, cell_state_out, output);

    // Check data types
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input,
                                                       input_to_forget_weights, input_to_cell_weights, input_to_output_weights,
                                                       recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights,
                                                       forget_gate_bias, cell_bias, output_gate_bias,
                                                       output_state_in, cell_state_in,
                                                       scratch_buffer, output_state_out, cell_state_out, output);

    // Check dimensions
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_to_forget_weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_to_cell_weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_to_output_weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_to_forget_weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_to_cell_weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_to_output_weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(forget_gate_bias->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_bias->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(output_gate_bias->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(output_state_in->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_state_in->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(scratch_buffer->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(output_state_out->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_state_out->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_bias->dimension(0) * 4 != scratch_buffer->dimension(0)
                                && cell_bias->dimension(0) * 3 != scratch_buffer->dimension(0));

    const unsigned int num_batches = input->dimension(1);
    const unsigned int num_cells   = input_to_output_weights->dimension(1);

    if(lstm_params.use_layer_norm())
    {
        // If CIFG is used, input layer normalization weights tensor is omitted
        if(lstm_params.has_cifg_opt())
        {
            ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.input_layer_norm_weights() != nullptr);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.input_layer_norm_weights());
            ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.input_layer_norm_weights()->num_dimensions() > 1);
            ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.input_layer_norm_weights()->dimension(0) != num_cells);
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, lstm_params.input_layer_norm_weights());
        }

        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.forget_layer_norm_weights(), lstm_params.cell_layer_norm_weights(), lstm_params.output_layer_norm_weights());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, lstm_params.forget_layer_norm_weights(), lstm_params.cell_layer_norm_weights(), lstm_params.output_layer_norm_weights());
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.forget_layer_norm_weights()->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_layer_norm_weights()->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.output_layer_norm_weights()->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.forget_layer_norm_weights()->dimension(0) != num_cells);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_layer_norm_weights()->dimension(0) != num_cells);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.output_layer_norm_weights()->dimension(0) != num_cells);
    }

    // Check peephole optimization
    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.cell_to_output_weights(), lstm_params.cell_to_forget_weights());
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_forget_weights()->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_output_weights()->num_dimensions() > 1);
    }

    TensorShape      units_out_transposed_shape = compute_transposed_shape(*recurrent_to_output_weights);
    TensorShape      num_units_transposed_shape = compute_transposed_shape(*forget_gate_bias);
    const TensorInfo units_out_transposed_info  = TensorInfo(units_out_transposed_shape, 1, input->data_type());
    const TensorInfo num_units_transposed_info  = TensorInfo(num_units_transposed_shape, 1, input->data_type());

    TensorInfo input_gate      = TensorInfo(TensorShape(num_cells, num_batches), 1, input->data_type());
    TensorInfo forget_gate     = TensorInfo(TensorShape(num_cells, num_batches), 1, input->data_type());
    TensorInfo output_gate_tmp = TensorInfo(TensorShape(num_cells, num_batches), 1, input->data_type());
    TensorInfo cell_state_tmp  = TensorInfo(TensorShape(num_cells, num_batches), 1, input->data_type());

    // Validate forget gate
    ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, input_to_forget_weights, (lstm_params.use_layer_norm()) ? nullptr : forget_gate_bias, &forget_gate));

    std::vector<const ITensorInfo *> inputs_vector;
    inputs_vector.emplace_back(input);
    inputs_vector.emplace_back(output_state_in);
    const TensorShape concat_shape       = arm_compute::misc::shape_calculator::calculate_concatenate_shape(inputs_vector, 0);
    TensorInfo        forget_gate_concat = TensorInfo(concat_shape, 1, input->data_type());

    ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(inputs_vector, &forget_gate_concat, Window::DimX));

    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(cell_state_in, lstm_params.cell_to_forget_weights(), &forget_gate, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN));
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&forget_gate, &forget_gate, &forget_gate, ConvertPolicy::SATURATE));
    }
    if(lstm_params.use_layer_norm())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLMeanStdDevNormalizationLayer::validate(&forget_gate));
        ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&forget_gate, lstm_params.forget_layer_norm_weights(), &forget_gate, 1, ConvertPolicy::SATURATE,
                                                                        RoundingPolicy::TO_NEAREST_EVEN));
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&forget_gate, forget_gate_bias, &forget_gate, ConvertPolicy::SATURATE));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&forget_gate, &forget_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));

    // Validate input gate
    if(!lstm_params.has_cifg_opt())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.input_to_input_weights(),
                                            lstm_params.recurrent_to_input_weights(),
                                            lstm_params.input_gate_bias());
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.input_to_input_weights()->num_dimensions() > 2);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.recurrent_to_input_weights()->num_dimensions() > 2);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.input_gate_bias()->num_dimensions() > 1);

        std::vector<const ITensorInfo *> lstm_weights;
        lstm_weights.emplace_back(lstm_params.input_to_input_weights());
        lstm_weights.emplace_back(lstm_params.recurrent_to_input_weights());
        TensorShape lstm_weights_concat_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(lstm_weights, 0);
        TensorInfo  lstm_gate_concat          = TensorInfo(lstm_weights_concat_shape, 1, input->data_type());
        ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(lstm_weights, &lstm_gate_concat, Window::DimX));

        ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, lstm_params.input_to_input_weights(), (lstm_params.use_layer_norm()) ? nullptr : lstm_params.input_gate_bias(), &input_gate));

        if(lstm_params.has_peephole_opt())
        {
            ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.cell_to_input_weights());
            ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_input_weights()->num_dimensions() > 1);
            ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(cell_state_in, lstm_params.cell_to_input_weights(), &input_gate, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN));
            ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&input_gate, &input_gate, &input_gate, ConvertPolicy::SATURATE));
        }

        if(lstm_params.use_layer_norm())
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLMeanStdDevNormalizationLayer::validate(&input_gate));
            ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&input_gate, lstm_params.input_layer_norm_weights(), &input_gate, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN));
            ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&input_gate, lstm_params.input_gate_bias(), &input_gate, ConvertPolicy::SATURATE));
        }
        ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&input_gate, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticSubtraction::validate(&forget_gate, &forget_gate, &forget_gate, ConvertPolicy::SATURATE));
    }

    // Validate cell state
    ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, input_to_cell_weights, (lstm_params.use_layer_norm()) ? nullptr : cell_bias, &cell_state_tmp));
    ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(output_state_in, &units_out_transposed_info, nullptr, &cell_state_tmp, 1.f, 0.f, GEMMInfo()));
    ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&cell_state_tmp, &cell_state_tmp, &cell_state_tmp, ConvertPolicy::SATURATE));
    if(lstm_params.use_layer_norm())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLMeanStdDevNormalizationLayer::validate(&cell_state_tmp));
        ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&cell_state_tmp, lstm_params.cell_layer_norm_weights(), &cell_state_tmp, 1, ConvertPolicy::SATURATE,
                                                                        RoundingPolicy::TO_NEAREST_EVEN));
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&cell_state_tmp, cell_bias, &cell_state_tmp, ConvertPolicy::SATURATE));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&cell_state_tmp, nullptr, activation_info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&cell_state_tmp, &input_gate, &cell_state_tmp, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN));
    ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&cell_state_tmp, &forget_gate, &cell_state_tmp, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN));
    ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&cell_state_tmp, &cell_state_tmp, &cell_state_tmp, ConvertPolicy::SATURATE));
    if(cell_threshold != 0.f)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&cell_state_tmp, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -cell_threshold,
                                                                                                              cell_threshold)));
    }

    std::vector<const ITensorInfo *> in_out_weights;
    in_out_weights.emplace_back(input_to_output_weights);
    in_out_weights.emplace_back(recurrent_to_output_weights);
    TensorShape in_out_weights_concat_shape = arm_compute::misc::shape_calculator::calculate_concatenate_shape(in_out_weights, 0);
    TensorInfo  in_out_gate_concat          = TensorInfo(in_out_weights_concat_shape, 1, input->data_type());
    ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(in_out_weights, &in_out_gate_concat, Window::DimX));
    // Validate output gate tmp
    ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(input, input_to_output_weights, (lstm_params.use_layer_norm()) ? nullptr : output_gate_bias, &output_gate_tmp));

    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&cell_state_tmp, lstm_params.cell_to_output_weights(), &output_gate_tmp, 1, ConvertPolicy::SATURATE,
                                                                        RoundingPolicy::TO_NEAREST_EVEN));
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&output_gate_tmp, &output_gate_tmp, &output_gate_tmp, ConvertPolicy::SATURATE));
    }
    if(lstm_params.use_layer_norm())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLMeanStdDevNormalizationLayer::validate(&output_gate_tmp));
        ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&output_gate_tmp, lstm_params.output_layer_norm_weights(), &output_gate_tmp, 1, ConvertPolicy::SATURATE,
                                                                        RoundingPolicy::TO_NEAREST_EVEN));
        ARM_COMPUTE_RETURN_ON_ERROR(CLArithmeticAddition::validate(&output_gate_tmp, output_gate_bias, &output_gate_tmp, ConvertPolicy::SATURATE));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&output_gate_tmp, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));

    // Validate output state
    ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(&cell_state_tmp, &cell_state_tmp, activation_info));
    ARM_COMPUTE_RETURN_ON_ERROR(CLPixelWiseMultiplication::validate(&cell_state_tmp, &output_gate_tmp, &output_gate_tmp, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_NEAREST_EVEN));
    if(lstm_params.has_projection())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayer::validate(&output_gate_tmp, lstm_params.projection_weights(), lstm_params.projection_bias(), output_state_out));
        if(projection_threshold != 0.f)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(CLActivationLayer::validate(output_state_out, output_state_out,
                                                                    ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -projection_threshold, projection_threshold)));
        }
    }

    // Validate copy kernel
    ARM_COMPUTE_RETURN_ON_ERROR(CLCopyKernel::validate(&cell_state_tmp, cell_state_out));
    ARM_COMPUTE_RETURN_ON_ERROR(CLCopyKernel::validate(output_state_out, output));

    // Validate scratch concatenation
    std::vector<const ITensorInfo *> inputs_vector_info_raw;
    if(!lstm_params.has_cifg_opt())
    {
        inputs_vector_info_raw.push_back(&input_gate);
    }
    inputs_vector_info_raw.push_back(&cell_state_tmp);
    inputs_vector_info_raw.push_back(&forget_gate);
    inputs_vector_info_raw.push_back(&output_gate_tmp);

    ARM_COMPUTE_RETURN_ON_ERROR(CLConcatenateLayer::validate(inputs_vector_info_raw, scratch_buffer, Window::DimX));
    return Status{};
}

void CLLSTMLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    _concat_inputs_forget_gate.run();

    _fully_connected_forget_gate.run();

    if(_run_peephole_opt)
    {
        _pixelwise_mul_forget_gate.run();
        _accum_forget_gate1.run();
    }
    if(_is_layer_norm_lstm)
    {
        _mean_std_norm_forget_gate.run();
        _pixelwise_mul_forget_gate_coeff.run();
        _accum_forget_gate_bias.run();
    }
    _activation_forget_gate.run();

    if(_run_cifg_opt)
    {
        CLScheduler::get().enqueue(*_ones_memset_kernel);
        _subtract_input_gate.run();
    }
    else
    {
        _fully_connected_input_gate.run();

        if(_run_peephole_opt)
        {
            _pixelwise_mul_input_gate.run();
            _accum_input_gate1.run();
        }

        if(_is_layer_norm_lstm)
        {
            _mean_std_norm_input_gate.run();
            _pixelwise_mul_input_gate_coeff.run();
            _accum_input_gate_bias.run();
        }
        _activation_input_gate.run();
    }

    _fully_connected_cell_state.run();
    CLScheduler::get().enqueue(*_transpose_cell_state);
    _gemm_cell_state1.run();
    _accum_cell_state1.run();
    if(_is_layer_norm_lstm)
    {
        _mean_std_norm_cell_gate.run();
        _pixelwise_mul_cell_gate_coeff.run();
        _accum_cell_gate_bias.run();
    }
    _activation_cell_state.run();
    _pixelwise_mul_cell_state1.run();
    _pixelwise_mul_cell_state2.run();
    _accum_cell_state2.run();

    if(_perform_cell_clipping)
    {
        _cell_clip.run();
    }

    _fully_connected_output.run();

    if(_run_peephole_opt)
    {
        _pixelwise_mul_output_state1.run();
        _accum_output1.run();
    }
    if(_is_layer_norm_lstm)
    {
        _mean_std_norm_output_gate.run();
        _pixelwise_mul_output_gate_coeff.run();
        _accum_output_gate_bias.run();
    }
    _activation_output.run();

    _activation_output_state.run();
    _pixelwise_mul_output_state2.run();

    if(_has_projection_weights)
    {
        _fully_connected_output_state.run();
        if(_perform_projection_clipping)
        {
            _projection_clip.run();
        }
    }

    CLScheduler::get().enqueue(*_copy_cell_state);
    CLScheduler::get().enqueue(*_copy_output);

    _concat_scratch_buffer.run();
}

void CLLSTMLayer::prepare()
{
    if(!_is_prepared)
    {
        _concat_weights_forget_gate.run();
        if(!_run_cifg_opt)
        {
            _concat_weights_input_gate.run();
        }
        _concat_weights_output.run();
        _is_prepared = true;
    }
}
} // namespace arm_compute
