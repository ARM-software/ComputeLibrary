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
#include "arm_compute/runtime/NEON/functions/NEQLSTMLayer.h"

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/InfoHelpers.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
using namespace arm_compute::utils::info_helpers;
namespace
{
Status validate_mm(GEMMLowpOutputStageInfo &gemmlowp_info, const ITensorInfo *mm_input, const ITensorInfo *mm_weights, const ITensorInfo *bias,
                   float gemmlowp_scale, const TensorInfo *mm_res_info, const TensorInfo *outstage_tensor_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyCore::validate(mm_input, mm_weights, nullptr, mm_res_info));
    ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(gemmlowp_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift));
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOutputStage::validate(mm_res_info, bias, outstage_tensor_info, gemmlowp_info));
    return Status{};
}
} // namespace

Status NEQLSTMLayer::TensorCopyKernel::validate(const ITensorInfo &src, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON(src.tensor_shape().num_dimensions() > max_dimension_supported);
    ARM_COMPUTE_RETURN_ERROR_ON(dst.tensor_shape().num_dimensions() > max_dimension_supported);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
    ARM_COMPUTE_RETURN_ERROR_ON(dst.tensor_shape().y() != src.tensor_shape().y());
    return Status{};
}

void NEQLSTMLayer::TensorCopyKernel::configure(ITensor &src, ITensor &dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(NEQLSTMLayer::TensorCopyKernel::validate(*src.info(), *dst.info()));
    _src      = &src;
    _dst      = &dst;
    _row_size = std::min(_src->info()->tensor_shape().x(), _dst->info()->tensor_shape().x());
    _window   = calculate_max_window(*_src->info(), Steps());
}

void NEQLSTMLayer::TensorCopyKernel::run()
{
    Iterator input_iter{ _src, _window };
    Iterator output_iter{ _dst, _window };

    execute_window_loop(_window, [&](const Coordinates &)
    {
        memcpy(output_iter.ptr(), input_iter.ptr(), _row_size);
    },
    input_iter, output_iter);
}

NEQLSTMLayer::NEQLSTMLayer(std::shared_ptr<IMemoryManager> memory_manager)
{
    _memory_group = MemoryGroup(std::move(memory_manager));
}

void NEQLSTMLayer::configure_mm(NEGEMMLowpMatrixMultiplyCore &mm, NEGEMMLowpOutputStage &outstage, GEMMLowpOutputStageInfo &gemmlowp_info,
                                const ITensor *mm_input, const ITensor *mm_weights, const ITensor *bias,
                                Tensor *mm_res, Tensor *outstage_res, float gemmlowp_scale,
                                const TensorInfo &mm_res_info, const TensorInfo &outstage_tensor_info)
{
    _memory_group.manage(mm_res);
    _memory_group.manage(outstage_res);

    mm_res->allocator()->init(mm_res_info);
    outstage_res->allocator()->init(outstage_tensor_info);

    // Configure matrix-multiplication
    mm.configure(mm_input, mm_weights, nullptr, mm_res);

    // Configure output stage
    quantization::calculate_quantized_multiplier(gemmlowp_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift);
    outstage.configure(mm_res, bias, outstage_res, gemmlowp_info);
    mm_res->allocator()->allocate();
}

void NEQLSTMLayer::configure(const ITensor *input,
                             const ITensor *input_to_forget_weights, const ITensor *input_to_cell_weights, const ITensor *input_to_output_weights,
                             const ITensor *recurrent_to_forget_weights, const ITensor *recurrent_to_cell_weights, const ITensor *recurrent_to_output_weights,
                             const ITensor *forget_gate_bias, const ITensor *cell_bias, const ITensor *output_gate_bias,
                             const ITensor *cell_state_in, const ITensor *output_state_in,
                             ITensor *cell_state_out, ITensor *output_state_out, ITensor *output,
                             const LSTMParams<ITensor> &lstm_params)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, input_to_forget_weights, input_to_cell_weights, input_to_output_weights,
                                 recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights,
                                 forget_gate_bias, cell_bias, output_gate_bias, cell_state_in, output_state_in, cell_state_out, output_state_out);

    // Set lstm parameters
    LSTMParams<ITensorInfo> lstm_params_info{};
    build_lstm_params_tensor_info(lstm_params, &lstm_params_info);

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(NEQLSTMLayer::validate(input->info(), input_to_forget_weights->info(), input_to_cell_weights->info(), input_to_output_weights->info(),
                                                      recurrent_to_forget_weights->info(), recurrent_to_cell_weights->info(), recurrent_to_output_weights->info(),
                                                      forget_gate_bias->info(), cell_bias->info(), output_gate_bias->info(),
                                                      cell_state_in->info(), output_state_in->info(), cell_state_out->info(), output_state_out->info(), output->info(),
                                                      lstm_params_info));

    const int batch_size  = input->info()->dimension(1);
    const int num_units   = input_to_output_weights->info()->dimension(1);
    const int output_size = output_state_out->info()->dimension(_out_state_output_size_dimension_idx);

    const UniformQuantizationInfo qinput           = input->info()->quantization_info().uniform();
    const UniformQuantizationInfo qcell_state_in   = cell_state_in->info()->quantization_info().uniform();
    const UniformQuantizationInfo qoutput_state_in = output_state_in->info()->quantization_info().uniform();

    _projection_bias             = lstm_params.projection_bias();
    _input_to_forget_weights     = input_to_forget_weights;
    _input_to_cell_weights       = input_to_cell_weights;
    _input_to_output_weights     = input_to_output_weights;
    _recurrent_to_forget_weights = recurrent_to_forget_weights;
    _recurrent_to_cell_weights   = recurrent_to_cell_weights;
    _recurrent_to_output_weights = recurrent_to_output_weights;
    _projection_weights          = lstm_params.projection_weights();

    // Layer normalization
    _has_layer_norm = lstm_params.use_layer_norm();
    if(_has_layer_norm)
    {
        set_layer_norm_weight(lstm_params.forget_layer_norm_weights(), LayerNormGate::Forget);
        set_layer_norm_weight(lstm_params.cell_layer_norm_weights(), LayerNormGate::Cell);
        set_layer_norm_weight(lstm_params.input_layer_norm_weights(), LayerNormGate::Input);
        set_layer_norm_weight(lstm_params.output_layer_norm_weights(), LayerNormGate::Output);

        set_layer_norm_bias(forget_gate_bias, LayerNormGate::Forget);
        set_layer_norm_bias(cell_bias, LayerNormGate::Cell);
        set_layer_norm_bias(lstm_params.input_gate_bias(), LayerNormGate::Input);
        set_layer_norm_bias(output_gate_bias, LayerNormGate::Output);
    }

    _has_cifg       = lstm_params.has_cifg_opt();
    _has_projection = lstm_params.has_projection();
    _has_peephole   = lstm_params.has_peephole_opt();

    // Calculate and decompose effective scales for optimizing matmul calculation
    const int32_t cell_shift = log2(qcell_state_in.scale);

    // Calculate quantized parameters for clipping.
    int16_t quantized_cell_clip = 0;
    if(lstm_params.cell_clip() > 0.0f)
    {
        quantized_cell_clip = quantize_qsymm16(lstm_params.cell_clip(), qcell_state_in);
    }
    _has_cell_clipping = quantized_cell_clip > 0;

    // Precompute effective bias for optimizing the matmul computations.
    if(!_has_cifg)
    {
        _input_to_input_weights     = lstm_params.input_to_input_weights();
        _recurrent_to_input_weights = lstm_params.recurrent_to_input_weights();

        _input_to_input_reduction.configure(_input_to_input_weights, &_input_to_input_eff_bias, GEMMLowpReductionKernelInfo(num_units, false, -qinput.offset, true));
        _recurrent_to_input_reduction.configure(_recurrent_to_input_weights, &_recurrent_to_input_eff_bias, GEMMLowpReductionKernelInfo(num_units, false, -qoutput_state_in.offset, true));
    }
    _input_to_forget_reduction.configure(input_to_forget_weights, &_input_to_forget_eff_bias, GEMMLowpReductionKernelInfo(num_units, false, -qinput.offset, true));
    _recurrent_to_forget_reduction.configure(recurrent_to_forget_weights, &_recurrent_to_forget_eff_bias, GEMMLowpReductionKernelInfo(num_units, false, -qoutput_state_in.offset, true));
    _input_to_cell_reduction.configure(input_to_cell_weights, &_input_to_cell_eff_bias, GEMMLowpReductionKernelInfo(num_units, false, -qinput.offset, true));
    _recurrent_to_cell_reduction.configure(recurrent_to_cell_weights, &_recurrent_to_cell_eff_bias, GEMMLowpReductionKernelInfo(num_units, false, -qoutput_state_in.offset, true));
    _input_to_output_reduction.configure(input_to_output_weights, &_input_to_output_eff_bias, GEMMLowpReductionKernelInfo(num_units, false, -qinput.offset, true));
    _recurrent_to_output_reduction.configure(recurrent_to_output_weights, &_recurrent_to_output_eff_bias, GEMMLowpReductionKernelInfo(num_units, false, -qoutput_state_in.offset, true));
    if(_has_projection)
    {
        _projection_reduction.configure(_projection_weights, &_projection_eff_bias, GEMMLowpReductionKernelInfo(output_size, false, lstm_params.hidden_state_zero(), true));
        if(_projection_bias != nullptr)
        {
            _projection_bias_add.configure(_projection_bias, &_projection_eff_bias, &_projection_eff_bias, ConvertPolicy::SATURATE);
        }
    }

    // Pre-transpose weights to be used in GEMM.
    _transpose_input_to_forget_weights.configure(input_to_forget_weights, &_input_to_forget_weights_transposed);
    _transpose_input_to_cell_weights.configure(input_to_cell_weights, &_input_to_cell_weights_transposed);
    _transpose_input_to_output_weights.configure(input_to_output_weights, &_input_to_output_weights_transposed);
    _transpose_recurrent_to_forget_weights.configure(recurrent_to_forget_weights, &_recurrent_to_forget_weights_transposed);
    _transpose_recurrent_to_cell_weights.configure(recurrent_to_cell_weights, &_recurrent_to_cell_weights_transposed);
    _transpose_recurrent_to_output_weights.configure(recurrent_to_output_weights, &_recurrent_to_output_weights_transposed);
    if(!_has_cifg)
    {
        _transpose_input_to_input_weights.configure(lstm_params.input_to_input_weights(), &_input_to_input_weights_transposed);
        _transpose_recurrent_to_input_weights.configure(lstm_params.recurrent_to_input_weights(), &_recurrent_to_input_weights_transposed);
    }
    if(_has_projection)
    {
        _transpose_projection_weights.configure(_projection_weights, &_projection_weights_transposed);
    }

    GEMMLowpOutputStageInfo gemmlowp_info;
    gemmlowp_info.type               = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_info.gemmlowp_min_bound = std::numeric_limits<int16_t>::lowest();
    gemmlowp_info.gemmlowp_max_bound = std::numeric_limits<int16_t>::max();
    gemmlowp_info.output_data_type   = DataType::QSYMM16;

    const TensorInfo mm_out_info(TensorShape(num_units, batch_size), 1, DataType::S32);
    // Forget gate.
    const TensorInfo forget_gate_outstage_info(mm_out_info.tensor_shape(), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.forget_intermediate_scale(), 0));
    const float      input_to_forget_scale = input_to_forget_weights->info()->quantization_info().uniform().scale * qinput.scale / lstm_params.forget_intermediate_scale();
    configure_mm(_mm_input_to_forget, _input_to_forget_outstage, gemmlowp_info,
                 input, &_input_to_forget_weights_transposed, &_input_to_forget_eff_bias,
                 &_mm_input_to_forget_res, &_input_to_forget_outstage_res, input_to_forget_scale,
                 mm_out_info, forget_gate_outstage_info);

    const float recurrent_to_forget_scale = recurrent_to_forget_weights->info()->quantization_info().uniform().scale * qoutput_state_in.scale / lstm_params.forget_intermediate_scale();
    configure_mm(_mm_recurrent_to_forget, _recurrent_to_forget_outstage, gemmlowp_info,
                 output_state_in, &_recurrent_to_forget_weights_transposed, &_recurrent_to_forget_eff_bias,
                 &_mm_recurrent_to_forget_res, &_recurrent_to_forget_outstage_res, recurrent_to_forget_scale,
                 mm_out_info, forget_gate_outstage_info);

    _accumulate_input_recurrent_forget.configure(&_input_to_forget_outstage_res, &_recurrent_to_forget_outstage_res, &_recurrent_to_forget_outstage_res, ConvertPolicy::SATURATE);
    _input_to_forget_outstage_res.allocator()->allocate();

    if(_has_peephole)
    {
        _mul_cell_to_forget_res.allocator()->init(TensorInfo(cell_state_in->info()->tensor_shape(), 1, DataType::S32));
        _memory_group.manage(&_mul_cell_to_forget_res);
        _pixelwise_mul_cell_to_forget.configure(cell_state_in, lstm_params.cell_to_forget_weights(), &_mul_cell_to_forget_res, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
        _cell_to_forget_outstage_res.allocator()->init(TensorInfo(_mul_cell_to_forget_res.info()->tensor_shape(), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.forget_intermediate_scale(), 0)));
        _memory_group.manage(&_cell_to_forget_outstage_res);
        const float cell_to_forget_scale = std::pow(2, cell_shift) * lstm_params.cell_to_forget_weights()->info()->quantization_info().uniform().scale / lstm_params.forget_intermediate_scale();
        quantization::calculate_quantized_multiplier(cell_to_forget_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift);
        _cell_to_forget_outstage.configure(&_mul_cell_to_forget_res, nullptr, &_cell_to_forget_outstage_res, gemmlowp_info);
        _mul_cell_to_forget_res.allocator()->allocate();
        _accumulate_cell_forget.configure(&_recurrent_to_forget_outstage_res, &_cell_to_forget_outstage_res, &_recurrent_to_forget_outstage_res, ConvertPolicy::SATURATE);
        _cell_to_forget_outstage_res.allocator()->allocate();
    }

    Tensor *forget_activation_input = &_recurrent_to_forget_outstage_res;

    if(_has_layer_norm)
    {
        configure_layer_norm(LayerNormGate::Forget, forget_activation_input);
        forget_activation_input->allocator()->allocate();
        forget_activation_input = &get_layer_norm_output(LayerNormGate::Forget);
    }

    // Output quantization info of Sigmoid and Tanh activations
    const QuantizationInfo sigmoid_tanh_outqinfo(1.f / 32768.f, 0);
    const TensorInfo       forget_gate_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, sigmoid_tanh_outqinfo);

    _memory_group.manage(&_forget_gate);
    _forget_gate.allocator()->init(forget_gate_info);
    _forget_gate_sigmoid.configure(forget_activation_input, &_forget_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    forget_activation_input->allocator()->allocate();

    // Modulation gate.
    const TensorInfo cell_outstage_info(mm_out_info.tensor_shape(), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.cell_intermediate_scale(), 0));
    const float      input_to_cell_scale = input_to_cell_weights->info()->quantization_info().uniform().scale * qinput.scale / lstm_params.cell_intermediate_scale();
    configure_mm(_mm_input_to_cell, _input_to_cell_outstage, gemmlowp_info,
                 input, &_input_to_cell_weights_transposed, &_input_to_cell_eff_bias,
                 &_mm_input_to_cell_res, &_input_to_cell_outstage_res, input_to_cell_scale,
                 mm_out_info, cell_outstage_info);

    const float recurrent_to_cell_scale = recurrent_to_cell_weights->info()->quantization_info().uniform().scale * qoutput_state_in.scale / lstm_params.cell_intermediate_scale();
    configure_mm(_mm_recurrent_to_cell, _recurrent_to_cell_outstage, gemmlowp_info,
                 output_state_in, &_recurrent_to_cell_weights_transposed, &_recurrent_to_cell_eff_bias,
                 &_mm_recurrent_to_cell_res, &_recurrent_to_cell_outstage_res, recurrent_to_cell_scale,
                 mm_out_info, cell_outstage_info);

    _accumulate_input_recurrent_modulation.configure(&_input_to_cell_outstage_res, &_recurrent_to_cell_outstage_res, &_recurrent_to_cell_outstage_res, ConvertPolicy::SATURATE);
    _input_to_cell_outstage_res.allocator()->allocate();

    Tensor *cell_activation_input = &_recurrent_to_cell_outstage_res;

    if(_has_layer_norm)
    {
        configure_layer_norm(LayerNormGate::Cell, cell_activation_input);
        cell_activation_input->allocator()->allocate();
        cell_activation_input = &get_layer_norm_output(LayerNormGate::Cell);
    }

    const TensorInfo cell_gate_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, sigmoid_tanh_outqinfo);

    _memory_group.manage(&_cell_gate);
    _cell_gate.allocator()->init(cell_gate_info);
    _cell_gate_tanh.configure(cell_activation_input, &_cell_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f));
    cell_activation_input->allocator()->allocate();

    // Input gate.
    const TensorInfo input_gate_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, sigmoid_tanh_outqinfo);
    _input_gate.allocator()->init(input_gate_info);
    _memory_group.manage(&_input_gate);
    if(_has_cifg)
    {
        _ones.allocator()->init(*_forget_gate.info());
        _input_gate_sub.configure(&_ones, &_forget_gate, &_input_gate, ConvertPolicy::SATURATE);
        _ones.allocator()->allocate();
    }
    else
    {
        const TensorInfo input_outstage_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.input_intermediate_scale(), 0));
        const float      input_to_input_scale = _input_to_input_weights->info()->quantization_info().uniform().scale * qinput.scale / lstm_params.input_intermediate_scale();
        configure_mm(_mm_input_to_input, _input_to_input_outstage, gemmlowp_info,
                     input, &_input_to_input_weights_transposed, &_input_to_input_eff_bias,
                     &_mm_input_to_input_res, &_input_to_input_outstage_res, input_to_input_scale,
                     mm_out_info, input_outstage_info);

        const float recurrent_to_input_scale = _recurrent_to_input_weights->info()->quantization_info().uniform().scale * qoutput_state_in.scale / lstm_params.input_intermediate_scale();
        configure_mm(_mm_recurrent_to_input, _recurrent_to_input_outstage, gemmlowp_info,
                     output_state_in, &_recurrent_to_input_weights_transposed, &_recurrent_to_input_eff_bias,
                     &_mm_recurrent_to_input_res, &_recurrent_to_input_outstage_res, recurrent_to_input_scale,
                     mm_out_info, input_outstage_info);
        _accumulate_input_recurrent_input.configure(&_input_to_input_outstage_res, &_recurrent_to_input_outstage_res, &_recurrent_to_input_outstage_res, ConvertPolicy::SATURATE);
        _input_to_input_outstage_res.allocator()->allocate();

        if(_has_peephole)
        {
            _mul_cell_to_input_res.allocator()->init(TensorInfo(cell_state_in->info()->tensor_shape(), 1, DataType::S32));
            _memory_group.manage(&_mul_cell_to_input_res);
            _pixelwise_mul_cell_to_input.configure(cell_state_in, lstm_params.cell_to_input_weights(), &_mul_cell_to_input_res, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
            const float cell_to_input_scale = std::pow(2, cell_shift) * lstm_params.cell_to_input_weights()->info()->quantization_info().uniform().scale / lstm_params.input_intermediate_scale();
            quantization::calculate_quantized_multiplier(cell_to_input_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift);
            _cell_to_input_outstage_res.allocator()->init(TensorInfo(_mul_cell_to_input_res.info()->tensor_shape(), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.input_intermediate_scale(), 0)));
            _memory_group.manage(&_cell_to_input_outstage_res);
            _cell_to_input_outstage.configure(&_mul_cell_to_input_res, nullptr, &_cell_to_input_outstage_res, gemmlowp_info);
            _mul_cell_to_input_res.allocator()->allocate();
            _accumulate_cell_input.configure(&_recurrent_to_input_outstage_res, &_cell_to_input_outstage_res, &_recurrent_to_input_outstage_res, ConvertPolicy::SATURATE);
            _cell_to_input_outstage_res.allocator()->allocate();
        }

        Tensor *input_activation_input = &_recurrent_to_input_outstage_res;

        if(_has_layer_norm)
        {
            configure_layer_norm(LayerNormGate::Input, input_activation_input);
            input_activation_input->allocator()->allocate();
            input_activation_input = &get_layer_norm_output(LayerNormGate::Input);
        }

        _input_gate_sigmoid.configure(input_activation_input, &_input_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
        input_activation_input->allocator()->allocate();
    }
    // Cell.
    // TODO(COMPMID-3395): Perform multiplication in the quantized domain in NEPixelWiseMultiplication
    _pixelwise_mul_forget_cell.configure(&_forget_gate, cell_state_in, &_forget_gate, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
    const float      cell_gate_scale      = _cell_gate.info()->quantization_info().uniform().scale;
    const float      mul_input_cell_scale = cell_gate_scale * std::pow(2, 15 + cell_shift);
    const TensorInfo mul_input_cell_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, QuantizationInfo(mul_input_cell_scale, 0));
    _memory_group.manage(&_mul_input_cell_res);
    _mul_input_cell_res.allocator()->init(mul_input_cell_info);
    _pixelwise_mul_input_cell.configure(&_input_gate, &_cell_gate, &_mul_input_cell_res, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
    _cell_gate.allocator()->allocate();
    _add_forget_cell.configure(&_forget_gate, &_mul_input_cell_res, cell_state_out, ConvertPolicy::SATURATE);
    _mul_input_cell_res.allocator()->allocate();
    _forget_gate.allocator()->allocate();
    if(_has_cell_clipping)
    {
        _cell_clip.configure(cell_state_out, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -quantized_cell_clip, quantized_cell_clip));
    }
    // Output gate.
    const TensorInfo output_outstage_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.output_intermediate_scale(), 0));
    const float      input_to_output_scale = input_to_output_weights->info()->quantization_info().uniform().scale * qinput.scale / lstm_params.output_intermediate_scale();
    configure_mm(_mm_input_to_output, _input_to_output_outstage, gemmlowp_info,
                 input, &_input_to_output_weights_transposed, &_input_to_output_eff_bias,
                 &_mm_input_to_output_res, &_input_to_output_outstage_res, input_to_output_scale,
                 mm_out_info, output_outstage_info);

    const float recurrent_to_output_scale = recurrent_to_output_weights->info()->quantization_info().uniform().scale * qoutput_state_in.scale / lstm_params.output_intermediate_scale();
    configure_mm(_mm_recurrent_to_output, _recurrent_to_output_outstage, gemmlowp_info,
                 output_state_in, &_recurrent_to_output_weights_transposed, &_recurrent_to_output_eff_bias,
                 &_mm_recurrent_to_output_res, &_recurrent_to_output_outstage_res, recurrent_to_output_scale,
                 mm_out_info, output_outstage_info);

    _accumulate_input_recurrent_output.configure(&_recurrent_to_output_outstage_res, &_input_to_output_outstage_res, &_recurrent_to_output_outstage_res, ConvertPolicy::SATURATE);
    _input_to_output_outstage_res.allocator()->allocate();

    if(_has_peephole)
    {
        // TODO(COMPMID-3395): Perform multiplication in the quantized domain in NEPixelWiseMultiplication
        // Here we are not using the output stage because all operations are done in float
        _mul_cell_to_output_res.allocator()->init(TensorInfo(cell_state_out->info()->tensor_shape(), 1, DataType::S32));
        _memory_group.manage(&_mul_cell_to_output_res);
        _pixelwise_mul_cell_to_output.configure(cell_state_out, lstm_params.cell_to_output_weights(), &_mul_cell_to_output_res, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);

        const float cell_to_output_scale = std::pow(2, cell_shift) * lstm_params.cell_to_output_weights()->info()->quantization_info().uniform().scale / lstm_params.output_intermediate_scale();
        quantization::calculate_quantized_multiplier(cell_to_output_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift);
        _cell_to_output_outstage_res.allocator()->init(TensorInfo(_mul_cell_to_output_res.info()->tensor_shape(), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.output_intermediate_scale(), 0)));
        _memory_group.manage(&_cell_to_output_outstage_res);
        _cell_to_output_outstage.configure(&_mul_cell_to_output_res, nullptr, &_cell_to_output_outstage_res, gemmlowp_info);
        _mul_cell_to_output_res.allocator()->allocate();

        _accumulate_cell_to_output.configure(&_recurrent_to_output_outstage_res, &_cell_to_output_outstage_res, &_recurrent_to_output_outstage_res, ConvertPolicy::SATURATE);
        _cell_to_output_outstage_res.allocator()->allocate();
    }

    Tensor *output_activation_input = &_recurrent_to_output_outstage_res;

    if(_has_layer_norm)
    {
        configure_layer_norm(LayerNormGate::Output, output_activation_input);
        output_activation_input->allocator()->allocate();
        output_activation_input = &get_layer_norm_output(LayerNormGate::Output);
    }
    const TensorInfo output_gate_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, sigmoid_tanh_outqinfo);

    _memory_group.manage(&_output_gate);
    _output_gate.allocator()->init(output_gate_info);
    _output_gate_sigmoid.configure(output_activation_input, &_output_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
    output_activation_input->allocator()->allocate();

    // Hidden.
    _hidden_tanh.configure(cell_state_out, &_input_gate, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f));
    // TODO(COMPMID-3395): Perform multiplication in the quantized domain in NEPixelWiseMultiplication
    _memory_group.manage(&_hidden_mul_res);
    const TensorInfo hidden_mul_res(_input_gate.info()->tensor_shape(), 1, DataType::S32);
    _hidden_mul_res.allocator()->init(hidden_mul_res);
    _pixelwise_mul_hidden.configure(&_output_gate, &_input_gate, &_hidden_mul_res, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
    _output_gate.allocator()->allocate();
    _input_gate.allocator()->allocate();
    const float hidden_state_scale = std::pow(2, -15) / lstm_params.hidden_state_scale() * std::pow(2, -15);
    quantization::calculate_quantized_multiplier(hidden_state_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift, /* ignore_epsilon */ true);
    gemmlowp_info.gemmlowp_offset  = lstm_params.hidden_state_zero();
    gemmlowp_info.output_data_type = output_state_in->info()->data_type();

    _projection_tensor_copy_required = (num_units != output_size);
    ITensor *hidden_gate_result      = output_state_out;

    _memory_group.manage(&_hidden_gate);

    if(_projection_tensor_copy_required)
    {
        _hidden_gate.allocator()->init(*output_state_out->info());
        _hidden_gate.info()->set_tensor_shape(_hidden_mul_res.info()->tensor_shape());
        hidden_gate_result = &_hidden_gate;
    }

    _hidden_outstage.configure(&_hidden_mul_res, nullptr, hidden_gate_result, gemmlowp_info);
    _hidden_mul_res.allocator()->allocate();

    // Projection.
    if(_has_projection)
    {
        const TensorInfo              projection_outstage_info(*output_state_out->info());
        const UniformQuantizationInfo qprojection      = _projection_weights->info()->quantization_info().uniform();
        const float                   projection_scale = qprojection.scale * lstm_params.hidden_state_scale() / qoutput_state_in.scale;
        gemmlowp_info.gemmlowp_offset                  = qoutput_state_in.offset;
        gemmlowp_info.gemmlowp_min_bound               = std::numeric_limits<int8_t>::lowest();
        gemmlowp_info.gemmlowp_max_bound               = std::numeric_limits<int8_t>::max();
        gemmlowp_info.output_data_type                 = DataType::QASYMM8_SIGNED;

        TensorInfo projection_mm_out_info{ mm_out_info };
        projection_mm_out_info.set_tensor_shape(TensorShape(output_size, batch_size));

        configure_mm(_mm_projection, _projection_outstage, gemmlowp_info,
                     hidden_gate_result, &_projection_weights_transposed, &_projection_eff_bias,
                     &_mm_projection_res, &_projection_outstage_res, projection_scale,
                     projection_mm_out_info, projection_outstage_info);

        ITensor *accumulate_destination = output_state_out;

        if(_projection_tensor_copy_required)
        {
            _hidden_gate.allocator()->allocate();
            _projection_accumulate_res.allocator()->init(*output_state_out->info());
            _projection_accumulate_res.info()->set_tensor_shape(_projection_outstage_res.info()->tensor_shape());
            _projection_output_to_accumulate_copy.configure(*output_state_out, _projection_accumulate_res);
            accumulate_destination = &_projection_accumulate_res;
        }

        _accumulate_projection.configure(&_projection_outstage_res, accumulate_destination, accumulate_destination, ConvertPolicy::SATURATE);
        _projection_outstage_res.allocator()->allocate();

        if(_projection_tensor_copy_required)
        {
            _projection_accumulate_to_output_copy.configure(_projection_accumulate_res, *output_state_out);
            _projection_accumulate_res.allocator()->allocate();
        }

        int8_t quantized_projection_clip{ 0 };
        if(lstm_params.projection_clip() > 0.0f)
        {
            quantized_projection_clip = utility::clamp<int8_t>(lstm_params.projection_clip() / qprojection.scale, -128, 127);
        }

        if(quantized_projection_clip > 0)
        {
            _projection_clip.configure(output_state_out, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -quantized_projection_clip, quantized_projection_clip));
            _has_projection_clipping = true;
        }
    }
    else
    {
        if(_projection_tensor_copy_required)
        {
            _hidden_to_output_copy.configure(_hidden_gate, *output_state_out);
            _hidden_gate.allocator()->allocate();
        }
    }

    // Copy output_state_out to output
    _copy_output.configure(output_state_out, output);
}

Status NEQLSTMLayer::validate(const ITensorInfo *input,
                              const ITensorInfo *input_to_forget_weights, const ITensorInfo *input_to_cell_weights, const ITensorInfo *input_to_output_weights,
                              const ITensorInfo *recurrent_to_forget_weights, const ITensorInfo *recurrent_to_cell_weights, const ITensorInfo *recurrent_to_output_weights,
                              const ITensorInfo *forget_gate_bias, const ITensorInfo *cell_bias, const ITensorInfo *output_gate_bias,
                              const ITensorInfo *cell_state_in, const ITensorInfo *output_state_in,
                              const ITensorInfo *cell_state_out, const ITensorInfo *output_state_out, const ITensorInfo *output,
                              const LSTMParams<ITensorInfo> &lstm_params)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, recurrent_to_forget_weights, recurrent_to_cell_weights,
                                        recurrent_to_output_weights, forget_gate_bias, cell_bias, output_gate_bias, cell_state_in, output_state_in,
                                        cell_state_out, output_state_out, output);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_dimensions() != 2, "Input must have exactly 2 dimensions");

    const unsigned int input_size  = input->dimension(0);
    const unsigned int batch_size  = input->dimension(1);
    const unsigned int num_units   = input_to_output_weights->dimension(1);
    const unsigned int output_size = output_state_out->dimension(_out_state_output_size_dimension_idx);

    ARM_COMPUTE_RETURN_ERROR_ON(input_to_output_weights->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(input_to_output_weights->dimension(0) != input_size);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input_to_output_weights, input_to_forget_weights, input_to_cell_weights);
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_to_output_weights->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(recurrent_to_output_weights->dimension(1) != num_units);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(recurrent_to_output_weights, recurrent_to_forget_weights, recurrent_to_cell_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_to_forget_weights, 1, DataType::QSYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_to_forget_weights, input_to_cell_weights, input_to_output_weights,
                                                       recurrent_to_forget_weights, recurrent_to_cell_weights, recurrent_to_output_weights);

    ARM_COMPUTE_RETURN_ERROR_ON(forget_gate_bias->num_dimensions() != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(forget_gate_bias->dimension(0) != num_units);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(forget_gate_bias, cell_bias, output_gate_bias);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(forget_gate_bias, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(forget_gate_bias, cell_bias, output_gate_bias);

    ARM_COMPUTE_RETURN_ERROR_ON(cell_state_in->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_state_in->dimension(0) != num_units);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_state_in->dimension(1) != batch_size);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(cell_state_in, 1, DataType::QSYMM16);

    ARM_COMPUTE_RETURN_ERROR_ON(output_state_in->num_dimensions() != 2);
    ARM_COMPUTE_RETURN_ERROR_ON(output_state_in->dimension(0) != output_size);
    ARM_COMPUTE_RETURN_ERROR_ON(output_state_in->dimension(1) != batch_size);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output_state_in);

    // Check whether peephole weights are all there or none
    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.cell_to_forget_weights(), lstm_params.cell_to_output_weights());
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lstm_params.cell_to_forget_weights(), 1, DataType::QSYMM16);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_forget_weights()->num_dimensions() != 1);
        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_to_forget_weights()->dimension(0) != num_units);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lstm_params.cell_to_forget_weights(), lstm_params.cell_to_output_weights());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(lstm_params.cell_to_forget_weights(), lstm_params.cell_to_output_weights());

        if(!lstm_params.has_cifg_opt())
        {
            ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.cell_to_input_weights());
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lstm_params.cell_to_forget_weights(), lstm_params.cell_to_input_weights());
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(lstm_params.cell_to_forget_weights(), lstm_params.cell_to_input_weights());
        }
    }

    const UniformQuantizationInfo qinput           = input->quantization_info().uniform();
    const UniformQuantizationInfo qcell_state_in   = cell_state_in->quantization_info().uniform();
    const UniformQuantizationInfo qoutput_state_in = output_state_in->quantization_info().uniform();

    // Calculate and decompose effective scales for optimizing matmul calculation
    const int32_t cell_shift = log2(qcell_state_in.scale);
    ARM_COMPUTE_RETURN_ERROR_ON(cell_shift > -9);

    // Calculate quantized parameters for clipping.
    int16_t quantized_cell_clip = 0;
    if(lstm_params.cell_clip() > 0.0f)
    {
        quantized_cell_clip = quantize_qsymm16(lstm_params.cell_clip(), qcell_state_in);
    }

    // Precompute effective bias for optimizing the matmul computations.
    const TensorInfo eff_bias_info(TensorShape(num_units), 1, DataType::S32);
    const TensorInfo projection_eff_bias_info(TensorShape(output_size), 1, DataType::S32);
    if(!lstm_params.has_cifg_opt())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(lstm_params.input_to_input_weights(), &eff_bias_info, GEMMLowpReductionKernelInfo(num_units, false, -qinput.offset, true)));
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(lstm_params.recurrent_to_input_weights(), &eff_bias_info, GEMMLowpReductionKernelInfo(num_units, false, -qoutput_state_in.offset,
                                                                               true)));
    }
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(input_to_forget_weights, &eff_bias_info, GEMMLowpReductionKernelInfo(num_units, false, -qinput.offset, true)));
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(recurrent_to_forget_weights, &eff_bias_info, GEMMLowpReductionKernelInfo(num_units, false, -qoutput_state_in.offset, true)));
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(input_to_cell_weights, &eff_bias_info, GEMMLowpReductionKernelInfo(num_units, false, -qinput.offset, true)));
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(recurrent_to_cell_weights, &eff_bias_info, GEMMLowpReductionKernelInfo(num_units, false, -qoutput_state_in.offset, true)));
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(input_to_output_weights, &eff_bias_info, GEMMLowpReductionKernelInfo(num_units, false, -qinput.offset, true)));
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(recurrent_to_output_weights, &eff_bias_info, GEMMLowpReductionKernelInfo(num_units, false, -qoutput_state_in.offset, true)));
    if(lstm_params.has_projection())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(lstm_params.projection_weights(), &projection_eff_bias_info, GEMMLowpReductionKernelInfo(output_size, false,
                                                                               lstm_params.hidden_state_zero(),
                                                                               true)));
        if(lstm_params.projection_bias() != nullptr)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lstm_params.projection_bias(), 1, DataType::S32);
            ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(lstm_params.projection_bias(), &projection_eff_bias_info, &projection_eff_bias_info, ConvertPolicy::SATURATE));
        }
    }

    const TensorInfo input_weights_transposed(TensorShape(num_units, input_size), 1, input_to_forget_weights->data_type(), input_to_forget_weights->quantization_info());
    const TensorInfo recurrent_weights_transposed(TensorShape(num_units, output_size), 1, recurrent_to_forget_weights->data_type(), recurrent_to_forget_weights->quantization_info());

    // Validate weights transpose
    ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(input_to_forget_weights, &input_weights_transposed));
    ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(input_to_cell_weights, &input_weights_transposed));
    ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(input_to_output_weights, &input_weights_transposed));
    ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(recurrent_to_forget_weights, &recurrent_weights_transposed));
    ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(recurrent_to_cell_weights, &recurrent_weights_transposed));
    ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(recurrent_to_output_weights, &recurrent_weights_transposed));
    if(!lstm_params.has_cifg_opt())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(lstm_params.input_to_input_weights(), &input_weights_transposed));
        ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(lstm_params.recurrent_to_input_weights(), &recurrent_weights_transposed));
    }
    if(lstm_params.has_projection())
    {
        const TensorInfo projection_weights_transposed(TensorShape(output_size, num_units), 1, lstm_params.projection_weights()->data_type(), lstm_params.projection_weights()->quantization_info());
        ARM_COMPUTE_RETURN_ON_ERROR(NETranspose::validate(lstm_params.projection_weights(), &projection_weights_transposed));
    }

    GEMMLowpOutputStageInfo gemmlowp_info;
    gemmlowp_info.type               = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_info.gemmlowp_min_bound = std::numeric_limits<int16_t>::lowest();
    gemmlowp_info.gemmlowp_max_bound = std::numeric_limits<int16_t>::max();
    gemmlowp_info.output_data_type   = DataType::QSYMM16;

    const bool has_layer_norm = lstm_params.use_layer_norm();

    // Forget gate.
    ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.forget_intermediate_scale() == 0);
    const TensorInfo forget_outstage_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.forget_intermediate_scale(), 0));
    const TensorInfo mm_out_info(TensorShape(num_units, batch_size), 1, DataType::S32);
    const float      input_to_forget_scale = input_to_forget_weights->quantization_info().uniform().scale * qinput.scale / lstm_params.forget_intermediate_scale();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, input, &input_weights_transposed, &eff_bias_info, input_to_forget_scale, &mm_out_info, &forget_outstage_info));

    const float recurrent_to_forget_scale = recurrent_to_forget_weights->quantization_info().uniform().scale * qoutput_state_in.scale / lstm_params.forget_intermediate_scale();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, output_state_in, &recurrent_weights_transposed, &eff_bias_info, recurrent_to_forget_scale, &mm_out_info, &forget_outstage_info));

    ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(&forget_outstage_info, &forget_outstage_info, &forget_outstage_info, ConvertPolicy::SATURATE));

    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lstm_params.cell_to_forget_weights(), 1, DataType::QSYMM16);
        ARM_COMPUTE_RETURN_ON_ERROR(NEPixelWiseMultiplication::validate(cell_state_in, lstm_params.cell_to_forget_weights(), &mm_out_info, 1.f, ConvertPolicy::SATURATE,
                                                                        RoundingPolicy::TO_ZERO));
        const float cell_to_forget_scale = std::pow(2, cell_shift) * lstm_params.cell_to_forget_weights()->quantization_info().uniform().scale / lstm_params.forget_intermediate_scale();
        ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(cell_to_forget_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift));
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOutputStage::validate(&mm_out_info, nullptr, &forget_outstage_info, gemmlowp_info));
        ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(&forget_outstage_info, &forget_outstage_info, &forget_outstage_info, ConvertPolicy::SATURATE));
    }

    if(has_layer_norm)
    {
        const ITensorInfo *w_info = lstm_params.forget_layer_norm_weights();
        const ITensorInfo *b_info = forget_gate_bias;
        ARM_COMPUTE_RETURN_ON_ERROR(validate_layer_norm(forget_outstage_info, *w_info, *b_info));
    }

    // Output quantization info of Sigmoid and Tanh activations
    const QuantizationInfo sigmoid_tanh_outqinfo(1.f / 32768.f, 0);
    const TensorInfo       forget_gate_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, sigmoid_tanh_outqinfo);

    ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(&forget_outstage_info, &forget_gate_info, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));

    // Modulation gate.
    ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.cell_intermediate_scale() == 0);
    const TensorInfo cell_outstage_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.cell_intermediate_scale(), 0));
    const float      input_to_cell_scale = input_to_cell_weights->quantization_info().uniform().scale * qinput.scale / lstm_params.cell_intermediate_scale();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, input, &input_weights_transposed, &eff_bias_info, input_to_cell_scale, &mm_out_info, &cell_outstage_info));

    const float recurrent_to_cell_scale = recurrent_to_cell_weights->quantization_info().uniform().scale * qoutput_state_in.scale / lstm_params.cell_intermediate_scale();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, output_state_in, &recurrent_weights_transposed, &eff_bias_info, recurrent_to_cell_scale, &mm_out_info, &cell_outstage_info));

    ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(&cell_outstage_info, &cell_outstage_info, &cell_outstage_info, ConvertPolicy::SATURATE));

    if(has_layer_norm)
    {
        const ITensorInfo *w_info = lstm_params.cell_layer_norm_weights();
        const ITensorInfo *b_info = cell_bias;
        ARM_COMPUTE_RETURN_ON_ERROR(validate_layer_norm(cell_outstage_info, *w_info, *b_info));
    }
    const TensorInfo cell_gate_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, sigmoid_tanh_outqinfo);

    ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(&cell_outstage_info, &cell_gate_info, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f)));

    // Input gate.
    const TensorInfo input_gate_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, sigmoid_tanh_outqinfo);
    if(lstm_params.has_cifg_opt())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(lstm_params.input_gate_bias() != nullptr, "Input gate bias must not be present when CIFG is used");
        ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticSubtraction::validate(&input_gate_info, &forget_gate_info, &forget_gate_info, ConvertPolicy::SATURATE));
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lstm_params.input_to_input_weights(), lstm_params.recurrent_to_input_weights(), lstm_params.input_gate_bias());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_to_forget_weights, lstm_params.input_to_input_weights(), lstm_params.recurrent_to_input_weights());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input_to_forget_weights, lstm_params.input_to_input_weights());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(recurrent_to_forget_weights, lstm_params.recurrent_to_input_weights());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(forget_gate_bias, lstm_params.input_gate_bias());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(forget_gate_bias, lstm_params.input_gate_bias());

        ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.input_intermediate_scale() == 0);
        const TensorInfo input_outstage_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.input_intermediate_scale(), 0));
        const float      input_to_input_scale = lstm_params.input_to_input_weights()->quantization_info().uniform().scale * qinput.scale / lstm_params.input_intermediate_scale();
        ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, input, &input_weights_transposed, &eff_bias_info, input_to_input_scale, &mm_out_info, &input_outstage_info));

        const float recurrent_to_input_scale = lstm_params.recurrent_to_input_weights()->quantization_info().uniform().scale * qoutput_state_in.scale / lstm_params.input_intermediate_scale();
        ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, output_state_in, &recurrent_weights_transposed, &eff_bias_info, recurrent_to_input_scale, &mm_out_info, &input_outstage_info));

        ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(&input_outstage_info, &input_outstage_info, &input_outstage_info, ConvertPolicy::SATURATE));

        if(lstm_params.has_peephole_opt())
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEPixelWiseMultiplication::validate(cell_state_in, lstm_params.cell_to_input_weights(), &mm_out_info, 1.f, ConvertPolicy::SATURATE,
                                                                            RoundingPolicy::TO_ZERO));
            const float cell_to_input_scale = std::pow(2, cell_shift) * lstm_params.cell_to_input_weights()->quantization_info().uniform().scale / lstm_params.input_intermediate_scale();
            ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(cell_to_input_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift));
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOutputStage::validate(&mm_out_info, &eff_bias_info, &input_outstage_info, gemmlowp_info));
            ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(&input_outstage_info, &input_outstage_info, &input_outstage_info, ConvertPolicy::SATURATE));
        }

        if(has_layer_norm)
        {
            const ITensorInfo *w_info = lstm_params.input_layer_norm_weights();
            const ITensorInfo *b_info = lstm_params.input_gate_bias();
            ARM_COMPUTE_RETURN_ON_ERROR(validate_layer_norm(input_outstage_info, *w_info, *b_info));
        }

        ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(&input_outstage_info, &input_gate_info, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f)));
    }
    // Cell.
    ARM_COMPUTE_RETURN_ON_ERROR(NEPixelWiseMultiplication::validate(&forget_gate_info, cell_state_in, &forget_gate_info, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO));
    ARM_COMPUTE_RETURN_ON_ERROR(NEPixelWiseMultiplication::validate(&input_gate_info, cell_state_in, &cell_gate_info, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO));
    ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(&forget_gate_info, &cell_gate_info, cell_state_out, ConvertPolicy::SATURATE));
    if(quantized_cell_clip > 0)
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(cell_state_out, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -quantized_cell_clip,
                                                                                                             quantized_cell_clip)));
    }
    // Output gate.
    ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.output_intermediate_scale() == 0);
    const TensorInfo output_outstage_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, QuantizationInfo(lstm_params.output_intermediate_scale(), 0));
    const float      input_to_output_scale = input_to_output_weights->quantization_info().uniform().scale * qinput.scale / lstm_params.output_intermediate_scale();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, input, &input_weights_transposed, &eff_bias_info, input_to_output_scale, &mm_out_info, &output_outstage_info));

    const float recurrent_to_output_scale = recurrent_to_output_weights->quantization_info().uniform().scale * qoutput_state_in.scale / lstm_params.output_intermediate_scale();
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, output_state_in, &recurrent_weights_transposed, &eff_bias_info, recurrent_to_output_scale, &mm_out_info, &output_outstage_info));

    ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(&output_outstage_info, &output_outstage_info, &output_outstage_info, ConvertPolicy::SATURATE));
    if(lstm_params.has_peephole_opt())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lstm_params.cell_to_output_weights(), 1, DataType::QSYMM16);
        // TODO(COMPMID-3395): Perform multiplication in the quantized domain in NEPixelWiseMultiplication
        // Here we are not using the output stage because all operations are done in float
        // const float cell_to_output_scale = std::pow(2, cell_shift) * lstm_params.cell_to_output_weights()->quantization_info().uniform().scale / lstm_params.output_intermediate_scale();
        // ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(cell_to_output_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift));
        ARM_COMPUTE_RETURN_ON_ERROR(NEPixelWiseMultiplication::validate(cell_state_out, lstm_params.cell_to_output_weights(), &output_outstage_info, 1.f, ConvertPolicy::SATURATE,
                                                                        RoundingPolicy::TO_ZERO));
        ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(&output_outstage_info, &output_outstage_info, &output_outstage_info, ConvertPolicy::SATURATE));
    }

    if(has_layer_norm)
    {
        const ITensorInfo *w_info = lstm_params.output_layer_norm_weights();
        const ITensorInfo *b_info = output_gate_bias;
        ARM_COMPUTE_RETURN_ON_ERROR(validate_layer_norm(output_outstage_info, *w_info, *b_info));
    }

    const TensorInfo output_gate_info(TensorShape(num_units, batch_size), 1, DataType::QSYMM16, sigmoid_tanh_outqinfo);
    ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(&output_outstage_info, &output_gate_info, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC)));

    // Hidden.
    ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(cell_state_out, &input_gate_info, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::TANH, 1.f, 1.f)));
    const TensorInfo hidden_mul_res(TensorShape(num_units, batch_size), 1, DataType::S32);
    const TensorInfo hidden_out_info(TensorShape(num_units, batch_size), 1, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ON_ERROR(NEPixelWiseMultiplication::validate(&output_gate_info, &input_gate_info, &hidden_mul_res, 1.f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO));

    ARM_COMPUTE_RETURN_ERROR_ON(lstm_params.hidden_state_scale() == 0);
    const float hidden_state_scale = std::pow(2, -15) / lstm_params.hidden_state_scale() * std::pow(2, -15);
    ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(hidden_state_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift, /* ignore_epsilon */ true));
    gemmlowp_info.gemmlowp_offset = lstm_params.hidden_state_zero();
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOutputStage::validate(&hidden_mul_res, nullptr, &hidden_out_info, gemmlowp_info));

    const bool projection_tensor_copy_required = num_units != output_size;

    // Projection.
    if(lstm_params.has_projection())
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(recurrent_to_forget_weights, lstm_params.projection_weights());
        ARM_COMPUTE_RETURN_ERROR_ON(qoutput_state_in.scale == 0);

        const UniformQuantizationInfo qprojection      = lstm_params.projection_weights()->quantization_info().uniform();
        const float                   projection_scale = qprojection.scale * lstm_params.hidden_state_scale() / qoutput_state_in.scale;
        ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(projection_scale, &gemmlowp_info.gemmlowp_multiplier, &gemmlowp_info.gemmlowp_shift));
        gemmlowp_info.gemmlowp_offset    = qoutput_state_in.offset;
        gemmlowp_info.gemmlowp_min_bound = std::numeric_limits<int8_t>::lowest();
        gemmlowp_info.gemmlowp_max_bound = std::numeric_limits<int8_t>::max();
        gemmlowp_info.output_data_type   = DataType::QASYMM8_SIGNED;

        const TensorInfo projection_outstage_info(*output_state_out);
        const TensorInfo projection_weights_transposed(TensorShape(output_size, num_units), 1, lstm_params.projection_weights()->data_type(), lstm_params.projection_weights()->quantization_info());

        TensorInfo projection_mm_out_info{ mm_out_info };
        projection_mm_out_info.set_tensor_shape(TensorShape(output_size, batch_size));

        ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemmlowp_info, &hidden_out_info, &projection_weights_transposed, &projection_eff_bias_info, projection_scale, &projection_mm_out_info,
                                                &projection_outstage_info));

        if(projection_tensor_copy_required)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEQLSTMLayer::TensorCopyKernel::validate(*output_state_out, projection_outstage_info));
        }

        ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAddition::validate(output_state_out, output_state_out, output_state_out, ConvertPolicy::SATURATE));

        if(projection_tensor_copy_required)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEQLSTMLayer::TensorCopyKernel::validate(projection_outstage_info, *output_state_out));
        }

        int8_t quantized_projection_clip{ 0 };
        if(lstm_params.projection_clip() > 0.0f)
        {
            quantized_projection_clip = quantize_qasymm8_signed(lstm_params.projection_clip(), qprojection);
        }

        if(quantized_projection_clip > 0)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(output_state_out, nullptr, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, -quantized_projection_clip,
                                                                                                                   quantized_projection_clip)));
        }
    }
    else
    {
        if(projection_tensor_copy_required)
        {
            ARM_COMPUTE_RETURN_ON_ERROR(NEQLSTMLayer::TensorCopyKernel::validate(hidden_out_info, *output_state_out));
        }
    }

    if(cell_state_out->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(cell_state_in, cell_state_out);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(cell_state_in, cell_state_out);
    }

    if(output_state_out->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output_state_out);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output_state_in, output_state_out);
    }

    ARM_COMPUTE_RETURN_ON_ERROR(NECopyKernel::validate(output_state_out, output));
    return Status{};
}

void NEQLSTMLayer::run()
{
    prepare();

    // Acquire all the temporaries
    MemoryGroupResourceScope scope_mg(_memory_group);

    // Forget gate.
    _mm_input_to_forget.run();
    _input_to_forget_outstage.run();

    _mm_recurrent_to_forget.run();
    _recurrent_to_forget_outstage.run();
    _accumulate_input_recurrent_forget.run();

    if(_has_peephole)
    {
        _pixelwise_mul_cell_to_forget.run();
        _cell_to_forget_outstage.run();
        _accumulate_cell_forget.run();
    }

    if(_has_layer_norm)
    {
        NEScheduler::get().schedule(&get_layer_norm(LayerNormGate::Forget), Window::DimY);
    }

    _forget_gate_sigmoid.run();

    // Modulation gate.
    _mm_input_to_cell.run();
    _input_to_cell_outstage.run();

    _mm_recurrent_to_cell.run();
    _recurrent_to_cell_outstage.run();
    _accumulate_input_recurrent_modulation.run();

    if(_has_layer_norm)
    {
        NEScheduler::get().schedule(&get_layer_norm(LayerNormGate::Cell), Window::DimY);
    }

    _cell_gate_tanh.run();

    // Input gate
    if(_has_cifg)
    {
        _input_gate_sub.run();
    }
    else
    {
        _mm_input_to_input.run();
        _input_to_input_outstage.run();
        _mm_recurrent_to_input.run();
        _recurrent_to_input_outstage.run();
        _accumulate_input_recurrent_input.run();

        if(_has_peephole)
        {
            _pixelwise_mul_cell_to_input.run();
            _cell_to_input_outstage.run();
            _accumulate_cell_input.run();
        }

        if(_has_layer_norm)
        {
            NEScheduler::get().schedule(&get_layer_norm(LayerNormGate::Input), Window::DimY);
        }

        _input_gate_sigmoid.run();
    }

    // Cell.
    _pixelwise_mul_forget_cell.run();
    _pixelwise_mul_input_cell.run();
    _add_forget_cell.run();

    if(_has_cell_clipping)
    {
        _cell_clip.run();
    }

    // Output gate.
    _mm_input_to_output.run();
    _input_to_output_outstage.run();
    _mm_recurrent_to_output.run();
    _recurrent_to_output_outstage.run();
    _accumulate_input_recurrent_output.run();
    if(_has_peephole)
    {
        _pixelwise_mul_cell_to_output.run();
        _cell_to_output_outstage.run();
        _accumulate_cell_to_output.run();
    }

    if(_has_layer_norm)
    {
        NEScheduler::get().schedule(&get_layer_norm(LayerNormGate::Output), Window::DimY);
    }

    _output_gate_sigmoid.run();

    // Hidden.
    _hidden_tanh.run();
    _pixelwise_mul_hidden.run();
    _hidden_outstage.run();

    // Projection.
    if(_has_projection)
    {
        _mm_projection.run();
        _projection_outstage.run();

        if(_projection_tensor_copy_required)
        {
            _projection_output_to_accumulate_copy.run();
        }

        _accumulate_projection.run();

        if(_projection_tensor_copy_required)
        {
            _projection_accumulate_to_output_copy.run();
        }

        if(_has_projection_clipping)
        {
            _projection_clip.run();
        }
    }
    else
    {
        if(_projection_tensor_copy_required)
        {
            _hidden_to_output_copy.run();
        }
    }

    // Copy output_state_out to output
    NEScheduler::get().schedule(&_copy_output, Window::DimY);
}

void NEQLSTMLayer::prepare()
{
    if(!_is_prepared)
    {
        // Pre-transpose weights to be used in GEMM.
        _input_to_forget_weights_transposed.allocator()->allocate();
        _input_to_cell_weights_transposed.allocator()->allocate();
        _input_to_output_weights_transposed.allocator()->allocate();
        _recurrent_to_forget_weights_transposed.allocator()->allocate();
        _recurrent_to_cell_weights_transposed.allocator()->allocate();
        _recurrent_to_output_weights_transposed.allocator()->allocate();
        _transpose_input_to_forget_weights.run();
        _transpose_input_to_cell_weights.run();
        _transpose_input_to_output_weights.run();
        _transpose_recurrent_to_forget_weights.run();
        _transpose_recurrent_to_cell_weights.run();
        _transpose_recurrent_to_output_weights.run();

        // Precompute effective biases
        if(_has_cifg)
        {
            std::fill_n(reinterpret_cast<int16_t *>(_ones.buffer()), _ones.info()->total_size() / _ones.info()->element_size(), 32767);
        }
        else
        {
            _input_to_input_eff_bias.allocator()->allocate();
            _recurrent_to_input_eff_bias.allocator()->allocate();
            NEScheduler::get().schedule(&_input_to_input_reduction, Window::DimY);
            NEScheduler::get().schedule(&_recurrent_to_input_reduction, Window::DimY);

            _input_to_input_weights_transposed.allocator()->allocate();
            _recurrent_to_input_weights_transposed.allocator()->allocate();
            _transpose_input_to_input_weights.run();
            _transpose_recurrent_to_input_weights.run();
            _input_to_input_weights->mark_as_unused();
            _recurrent_to_input_weights->mark_as_unused();
        }
        _input_to_forget_eff_bias.allocator()->allocate();
        _recurrent_to_forget_eff_bias.allocator()->allocate();
        _input_to_cell_eff_bias.allocator()->allocate();
        _recurrent_to_cell_eff_bias.allocator()->allocate();
        _input_to_output_eff_bias.allocator()->allocate();
        _recurrent_to_output_eff_bias.allocator()->allocate();
        NEScheduler::get().schedule(&_input_to_forget_reduction, Window::DimY);
        NEScheduler::get().schedule(&_recurrent_to_forget_reduction, Window::DimY);
        NEScheduler::get().schedule(&_input_to_cell_reduction, Window::DimY);
        NEScheduler::get().schedule(&_recurrent_to_cell_reduction, Window::DimY);
        NEScheduler::get().schedule(&_input_to_output_reduction, Window::DimY);
        NEScheduler::get().schedule(&_recurrent_to_output_reduction, Window::DimY);

        if(_has_projection)
        {
            _projection_eff_bias.allocator()->allocate();
            NEScheduler::get().schedule(&_projection_reduction, Window::DimY);
            if(_projection_bias != nullptr)
            {
                _projection_bias_add.run();
                _projection_bias->mark_as_unused();
            }

            _projection_weights_transposed.allocator()->allocate();
            _transpose_projection_weights.run();
            _projection_weights->mark_as_unused();

            if(!_projection_tensor_copy_required)
            {
                _hidden_gate.mark_as_unused();
                _projection_accumulate_res.mark_as_unused();
            }
        }

        // Mark weights as unused
        _input_to_forget_weights->mark_as_unused();
        _input_to_cell_weights->mark_as_unused();
        _input_to_output_weights->mark_as_unused();
        _recurrent_to_forget_weights->mark_as_unused();
        _recurrent_to_cell_weights->mark_as_unused();
        _recurrent_to_output_weights->mark_as_unused();

        _is_prepared = true;
    }
}

} // namespace arm_compute
