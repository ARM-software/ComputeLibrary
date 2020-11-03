/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NEConvertFullyConnectedWeightsKernel.h"
#include "src/core/NEON/kernels/NEConvertQuantizedSignednessKernel.h"
#include "src/core/NEON/kernels/NEFlattenLayerKernel.h"
#include "src/core/NEON/kernels/NEFlattenLayerKernel.h"
#include "src/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpReductionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "src/core/NEON/kernels/NETransposeKernel.h"

#include "support/MemorySupport.h"

#include <algorithm>
#include <cmath>

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;

namespace
{
// Get min, max bound of a quantized assymetric output tensor, with the effect of fused activation
std::pair<PixelValue, PixelValue> get_quantized_asymmetric_output_min_max(const QuantizationInfo &q_info, const ActivationLayerInfo &act_info, DataType data_type)
{
    PixelValue type_min{};
    PixelValue type_max{};
    std::tie(type_min, type_max) = get_min_max(data_type);
    const UniformQuantizationInfo q_unif = q_info.uniform();

    if(act_info.enabled())
    {
        switch(act_info.activation())
        {
            case ActivationLayerInfo::ActivationFunction::RELU:
                type_min = PixelValue(q_unif.offset);
                break;
            case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                type_min = PixelValue(q_unif.offset);
                type_max = PixelValue(act_info.a(), data_type, q_info);
                break;
            case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                type_min = PixelValue(act_info.b(), data_type, q_info);
                type_max = PixelValue(act_info.a(), data_type, q_info);
                break;
            default:
                ARM_COMPUTE_ERROR("Activation function not supported.");
                break;
        }
    }

    return std::make_pair(type_min, type_max);
}

Status get_gemmlowp_output_stage_info(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const ActivationLayerInfo &act,
                                      GEMMLowpOutputStageInfo &gemmlowp_output_stage_info)
{
    const auto                    data_type = input->data_type();
    const QuantizationInfo        oq_info   = output->quantization_info();
    const UniformQuantizationInfo iq_unif   = input->quantization_info().uniform();
    const UniformQuantizationInfo wq_unif   = weights->quantization_info().uniform();
    const UniformQuantizationInfo oq_unif   = oq_info.uniform();

    float   multiplier = (iq_unif.scale * wq_unif.scale) / oq_unif.scale;
    int32_t output_multiplier;
    int32_t output_shift;

    ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));

    PixelValue type_min{};
    PixelValue type_max{};
    std::tie(type_min, type_max) = get_quantized_asymmetric_output_min_max(oq_info, act, data_type);

    gemmlowp_output_stage_info.gemmlowp_multiplier = output_multiplier;
    gemmlowp_output_stage_info.gemmlowp_shift      = output_shift;
    gemmlowp_output_stage_info.gemmlowp_offset     = oq_unif.offset;
    gemmlowp_output_stage_info.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_output_stage_info.gemmlowp_min_bound  = type_min.get<int32_t>();
    gemmlowp_output_stage_info.gemmlowp_max_bound  = type_max.get<int32_t>();

    return Status{};
}

Status validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const ActivationLayerInfo &act)
{
    if(is_data_type_quantized_asymmetric(input->data_type()))
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info(input->quantization_info().uniform().scale, -input->quantization_info().uniform().offset);
        const QuantizationInfo weights_quantization_info(weights->quantization_info().uniform().scale, -weights->quantization_info().uniform().offset);

        GEMMLowpOutputStageInfo gemmlowp_output_stage_info;
        ARM_COMPUTE_RETURN_ON_ERROR(get_gemmlowp_output_stage_info(input, weights, output, act, gemmlowp_output_stage_info));

        GEMMInfo gemm_info;
        gemm_info.set_gemmlowp_output_stage(gemmlowp_output_stage_info);

        // Validate gemmlowp function
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyCore::validate(&input->clone()->set_quantization_info(input_quantization_info),
                                                                           &weights->clone()->set_quantization_info(weights_quantization_info),
                                                                           biases,
                                                                           output,
                                                                           gemm_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMM::validate(input, weights, biases, output, 1.f, 1.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run */)));
    }

    return Status{};
}
} // namespace

void NEFullyConnectedLayerReshapeWeights::configure(const ITensor *input, ITensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<NETransposeKernel>();
    k->configure(input, output);
    _kernel = std::move(k);
}

Status NEFullyConnectedLayerReshapeWeights::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return NETransposeKernel::validate(input, output);
}

NEFullyConnectedLayer::~NEFullyConnectedLayer() = default;

NEFullyConnectedLayer::NEFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager, IWeightsManager *weights_manager)
    : _memory_group(std::move(memory_manager)), _weights_manager(weights_manager), _flatten_kernel(), _convert_weights(), _convert_weights_managed(), _reshape_weights_function(),
      _reshape_weights_managed_function(), _mm_gemm(nullptr, weights_manager), _mm_gemmlowp(nullptr, weights_manager), _flatten_output(), _converted_weights_output(), _reshape_weights_output(),
      _original_weights(nullptr), _are_weights_converted(true), _are_weights_reshaped(false), _is_fc_after_conv(false), _is_quantized_asymmetric(false), _is_prepared(false)
{
}

void NEFullyConnectedLayer::configure_mm(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act)
{
    if(_is_quantized_asymmetric)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = input->info()->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->info()->quantization_info();

        input->info()->set_quantization_info(QuantizationInfo(input_quantization_info.uniform().scale, -input_quantization_info.uniform().offset));
        weights->info()->set_quantization_info(QuantizationInfo(weights_quantization_info.uniform().scale, -weights_quantization_info.uniform().offset));

        // Configure gemmlowp function and output stage for asymmetric quantized types
        GEMMLowpOutputStageInfo gemmlowp_output_stage_info;
        const Status            status = get_gemmlowp_output_stage_info(input->info(), weights->info(), output->info(), act, gemmlowp_output_stage_info);
        ARM_COMPUTE_ERROR_ON(status.error_code() != ErrorCode::OK);

        GEMMInfo gemm_info;
        gemm_info.set_gemmlowp_output_stage(gemmlowp_output_stage_info);
        gemm_info.set_activation_info(act);
        _mm_gemmlowp.configure(input, weights, biases, output, gemm_info);

        // Revert back QuantizatioInfo as input and weights could be used in other fully connected layers
        input->info()->set_quantization_info(input_quantization_info);
        weights->info()->set_quantization_info(weights_quantization_info);
    }
    else
    {
        // Configure matrix multiply kernel
        GEMMInfo gemm_info(false, false, true /* Reshape weights only for the first run */);
        gemm_info.set_activation_info(act);
        _mm_gemm.configure(input, weights, biases, output, 1.f, 1.0f, gemm_info);
    }
}

void NEFullyConnectedLayer::configure_conv_fc(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act)
{
    ARM_COMPUTE_ERROR_ON((weights->info()->dimension(1) != (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))));

    // If the fully connected layer is called after a convolution layer, the input tensor must be linearized

    // Initialize output tensor for flatten
    TensorShape shape_flatten = compute_flatten_shape(input->info());
    _flatten_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_flatten));

    // Configure flatten kernel
    _memory_group.manage(&_flatten_output);

    _flatten_kernel = arm_compute::support::cpp14::make_unique<NEFlattenLayerKernel>();
    _flatten_kernel->configure(input, &_flatten_output);

    // Configure matrix multiply kernel
    configure_mm(&_flatten_output, weights, biases, output, act);

    // Allocate the output tensor for flatten once all the configure methods have been called
    _flatten_output.allocator()->allocate();
}

void NEFullyConnectedLayer::configure_fc_fc(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act)
{
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != weights->info()->dimension(1));

    // Configure matrix multiply kernel
    configure_mm(input, weights, biases, output, act);
}

void NEFullyConnectedLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output,
                                      FullyConnectedLayerInfo fc_info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEFullyConnectedLayer::validate(input->info(),
                                                               weights->info(),
                                                               biases != nullptr ? biases->info() : nullptr,
                                                               output->info(),
                                                               fc_info));

    _are_weights_converted   = true;
    _are_weights_reshaped    = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
    _is_fc_after_conv        = true;
    _is_quantized_asymmetric = is_data_type_quantized_asymmetric(input->info()->data_type());
    _original_weights        = weights;

    if(_weights_manager)
    {
        _weights_manager->manage(weights);
    }

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    const ITensor *weights_to_use = weights;

    // Check if we have a fully connected layer with batches
    const bool is_batched_fc_layer = output->info()->dimension(1) > 1;
    if(is_batched_fc_layer)
    {
        _is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) && (std::equal(input->info()->tensor_shape().cbegin() + 3,
                                                                                  input->info()->tensor_shape().cend(),
                                                                                  output->info()->tensor_shape().cbegin() + 1));
    }
    else
    {
        _is_fc_after_conv = input->info()->num_dimensions() > 1;
    }

    // Reshape weights if needed
    if(!_are_weights_reshaped)
    {
        if(_weights_manager && _weights_manager->are_weights_managed(weights))
        {
            _reshape_weights_managed_function.configure(weights);
            weights_to_use = _weights_manager->acquire(weights, &_reshape_weights_managed_function);
        }
        else
        {
            // Reshape the weights
            _reshape_weights_function.configure(weights, &_reshape_weights_output);
            weights_to_use = &_reshape_weights_output;
        }
    }

    // Convert weights if needed
    if(_is_fc_after_conv && (input->info()->data_layout() != fc_info.weights_trained_layout))
    {
        if(_weights_manager && _weights_manager->are_weights_managed(weights_to_use))
        {
            _convert_weights_managed.configure(weights_to_use,
                                               input->info()->tensor_shape(),
                                               fc_info.weights_trained_layout);
            weights_to_use = _weights_manager->acquire(weights, &_convert_weights_managed);
        }
        else
        {
            // Convert weights
            _convert_weights.configure(weights_to_use,
                                       &_converted_weights_output,
                                       input->info()->tensor_shape(),
                                       fc_info.weights_trained_layout);

            weights_to_use = &_converted_weights_output;
        }
        _are_weights_converted = false;
    }

    if(_is_fc_after_conv)
    {
        // Fully Connected layer after a Convolution Layer without batches
        configure_conv_fc(input, weights_to_use, biases, output, fc_info.activation_info);
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        configure_fc_fc(input, weights_to_use, biases, output, fc_info.activation_info);
    }

    _are_weights_reshaped = _are_weights_reshaped || fc_info.retain_internal_weights;
}

Status NEFullyConnectedLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                       FullyConnectedLayerInfo fc_info)
{
    ARM_COMPUTE_UNUSED(fc_info.retain_internal_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(biases != nullptr && biases->num_dimensions() > 1);

    bool weights_reshaped = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
    bool is_fc_after_conv = true;

    const ITensorInfo &flatten_input     = TensorInfo(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_flatten_shape(input)));
    const ITensorInfo &reshaped_weights  = TensorInfo(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_transposed_shape(*weights)));
    const ITensorInfo &converted_weights = weights_reshaped ? TensorInfo(weights->clone()->set_is_resizable(true).reset_padding()) : TensorInfo(*reshaped_weights.clone());

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    const ITensorInfo *input_to_use   = input;
    const ITensorInfo *weights_to_use = weights;

    // Check if we have a fully connected layer with batches
    const bool is_batched_fc_layer = output->dimension(1) > 1;

    if(is_batched_fc_layer)
    {
        is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) && (std::equal(input->tensor_shape().cbegin() + 3,
                                                                                 input->tensor_shape().cend(),
                                                                                 output->tensor_shape().cbegin() + 1));
    }
    else
    {
        is_fc_after_conv = input->num_dimensions() > 1;
    }

    if(!weights_reshaped)
    {
        // Validate reshape weights kernel
        ARM_COMPUTE_RETURN_ON_ERROR(NEFullyConnectedLayerReshapeWeights::validate(weights, &reshaped_weights));
        weights_to_use = &reshaped_weights;
    }

    if(is_fc_after_conv && (input->data_layout() != fc_info.weights_trained_layout))
    {
        // Validate convert weights kernel
        ARM_COMPUTE_RETURN_ON_ERROR(NEConvertFullyConnectedWeights::validate(weights_to_use,
                                                                             &converted_weights,
                                                                             input->tensor_shape(),
                                                                             fc_info.weights_trained_layout));
        weights_to_use = &converted_weights;
    }

    if(is_fc_after_conv)
    {
        // Fully Connected layer after a Convolution Layer without batches
        ARM_COMPUTE_RETURN_ERROR_ON((weights_to_use->dimension(1) != (input->dimension(0) * input->dimension(1) * input->dimension(2))));

        // Validate flatten kernel
        ARM_COMPUTE_RETURN_ON_ERROR(NEFlattenLayerKernel::validate(input, &flatten_input));
        input_to_use = &flatten_input;
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != weights_to_use->dimension(1));
    }
    // Validate matrix multiply kernel
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(input_to_use, weights_to_use, biases, output, fc_info.activation_info));

    return Status{};
}

void NEFullyConnectedLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Linearize input if it comes from a convolutional layer
    if(_is_fc_after_conv)
    {
        NEScheduler::get().schedule(_flatten_kernel.get(), Window::DimY);
    }

    // Run matrix multiply
    if(_is_quantized_asymmetric)
    {
        _mm_gemmlowp.run();
    }
    else
    {
        _mm_gemm.run();
    }
}

void NEFullyConnectedLayer::prepare()
{
    if(!_is_prepared)
    {
        if(!_weights_manager)
        {
            ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());
        }

        auto release_unused = [](Tensor * w)
        {
            if(!w->is_used())
            {
                w->allocator()->free();
            }
        };

        // Pointer to current weights
        const ITensor *cur_weights = _original_weights;

        // Reshape of the weights (happens only once)
        if(!_are_weights_reshaped)
        {
            if(_weights_manager && _weights_manager->are_weights_managed(_original_weights))
            {
                cur_weights = _weights_manager->run(cur_weights, &_reshape_weights_managed_function);
            }
            else
            {
                // Reshape of the weights (happens only once)
                if(!_are_weights_reshaped)
                {
                    // Run reshape weights kernel and mark weights as unused
                    _reshape_weights_output.allocator()->allocate();
                    _reshape_weights_function.run();
                }
                cur_weights->mark_as_unused();
                cur_weights = &_reshape_weights_output;
            }
            _are_weights_reshaped = true;
        }

        // Convert weights if needed (happens only once)
        if(!_are_weights_converted)
        {
            if(_weights_manager && _weights_manager->are_weights_managed(cur_weights))
            {
                _weights_manager->run(cur_weights, &_convert_weights_managed);
            }
            else
            {
                _converted_weights_output.allocator()->allocate();
                _convert_weights.run();
                cur_weights->mark_as_unused();
            }

            _are_weights_converted = true;
        }

        // Release reshaped weights if unused
        release_unused(&_reshape_weights_output);

        // Prepare GEMM prepare and release unused weights
        if(!_is_quantized_asymmetric)
        {
            _mm_gemm.prepare();
        }

        // Release converted weights if unused
        release_unused(&_reshape_weights_output);
        release_unused(&_converted_weights_output);

        _is_prepared = true;
    }
}
} // namespace arm_compute
