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
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
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
#include "src/core/CL/kernels/CLTransposeKernel.h"
#include "support/Cast.h"
#include "support/MemorySupport.h"

#include <algorithm>

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::utils::cast;

namespace
{
Status construct_gemmlowp_output_stage(const ITensorInfo &input, const ITensorInfo &weights, const ITensorInfo &output,
                                       GEMMLowpOutputStageInfo &gemmlowp_output_stage, ActivationLayerInfo activation_info)
{
    gemmlowp_output_stage.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_output_stage.gemmlowp_offset     = 0;
    gemmlowp_output_stage.gemmlowp_multiplier = 0;
    gemmlowp_output_stage.gemmlowp_shift      = 0;

    const auto data_type = input.data_type();

    // Configure output stage for quantized case
    if(is_data_type_quantized_asymmetric(data_type))
    {
        const QuantizationInfo        oq_info = output.quantization_info();
        const UniformQuantizationInfo iq_unif = input.quantization_info().uniform();
        const UniformQuantizationInfo wq_unif = weights.quantization_info().uniform();
        const UniformQuantizationInfo oq_unif = oq_info.uniform();

        const auto output_quant_info = (output.total_size() == 0) ? iq_unif : oq_unif;

        const float multiplier        = (iq_unif.scale * wq_unif.scale) / output_quant_info.scale;
        int         output_multiplier = 0;
        int         output_shift      = 0;
        ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));

        PixelValue type_min{};
        PixelValue type_max{};
        std::tie(type_min, type_max) = get_min_max(data_type);

        if(activation_info.enabled())
        {
            std::tie(type_min, type_max) = get_quantized_activation_min_max(activation_info, data_type, output_quant_info);
        }

        // Set the GEMMLowp output stage info
        gemmlowp_output_stage.gemmlowp_offset     = output_quant_info.offset;
        gemmlowp_output_stage.gemmlowp_multiplier = output_multiplier;
        gemmlowp_output_stage.gemmlowp_shift      = output_shift;
        gemmlowp_output_stage.gemmlowp_multipliers.push_back(output_multiplier);
        gemmlowp_output_stage.gemmlowp_shifts.push_back(output_shift);
        type_min.get(gemmlowp_output_stage.gemmlowp_min_bound);
        type_max.get(gemmlowp_output_stage.gemmlowp_max_bound);
    }

    return Status{};
}

Status validate_mm(const ITensorInfo &input, const ITensorInfo &weights, const ITensorInfo *bias, const ITensorInfo &output, const FullyConnectedLayerInfo &fc_info)
{
    GEMMLowpOutputStageInfo gemmlowp_output_stage;
    ARM_COMPUTE_RETURN_ON_ERROR(construct_gemmlowp_output_stage(input, weights, output, gemmlowp_output_stage, fc_info.activation_info));

    const GEMMInfo &gemm_info = GEMMInfo(false,                           // is_a_reshaped
                                         false,                           // is_b_reshaped
                                         true,                            // reshape_b_only_on_first_run
                                         0,                               // depth_output_gemm3d
                                         false,                           // reinterpret_input_as_3d
                                         fc_info.retain_internal_weights, // retain_internal_weights
                                         gemmlowp_output_stage,           // gemmlowp_output_stage
                                         fc_info.fp_mixed_precision,      // fp_mixed_precision
                                         true,                            // broadcast_bias
                                         ActivationLayerInfo());          // activation_info

    if(is_data_type_quantized_asymmetric(input.data_type()))
    {
        const UniformQuantizationInfo iq_info = input.quantization_info().uniform();
        const UniformQuantizationInfo wq_info = weights.quantization_info().uniform();

        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info(iq_info.scale, -iq_info.offset);
        const QuantizationInfo weights_quantization_info(wq_info.scale, -wq_info.offset);

        // Validate gemmlowp function
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMMLowpMatrixMultiplyCore::validate(&input.clone()->set_quantization_info(input_quantization_info),
                                                                           &weights.clone()->set_quantization_info(weights_quantization_info),
                                                                           bias,
                                                                           &output,
                                                                           gemm_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CLGEMM::validate(&input, &weights, bias, &output, 1.f, 1.f, gemm_info));
    }

    return Status{};
}
} // namespace

void CLFullyConnectedLayerReshapeWeights::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLFullyConnectedLayerReshapeWeights::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLTransposeKernel>();
    k->configure(compile_context, input, output);
    _kernel = std::move(k);
}

Status CLFullyConnectedLayerReshapeWeights::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    return CLTransposeKernel::validate(input, output);
}

CLFullyConnectedLayer::CLFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager, IWeightsManager *weights_manager)
    : _memory_group(memory_manager), _weights_manager(weights_manager), _convert_weights(), _convert_weights_managed(), _reshape_weights_managed_function(), _flatten_layer(), _reshape_weights_function(),
      _mm_gemm(memory_manager, weights_manager), _mm_gemmlowp(memory_manager), _flatten_output(), _converted_weights_output(), _reshape_weights_output(), _are_weights_converted(true),
      _are_weights_reshaped(true), _is_fc_after_conv(true), _is_quantized(false), _is_prepared(false), _original_weights(nullptr)
{
}
void CLFullyConnectedLayer::configure_mm(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output,
                                         const FullyConnectedLayerInfo &fc_info)
{
    GEMMLowpOutputStageInfo gemmlowp_output_stage;
    construct_gemmlowp_output_stage(*input->info(), *weights->info(), *output->info(), gemmlowp_output_stage, fc_info.activation_info);

    const GEMMInfo &gemm_info = GEMMInfo(false,                           // is_a_reshaped
                                         false,                           // is_b_reshaped
                                         true,                            // reshape_b_only_on_first_run
                                         0,                               // depth_output_gemm3d
                                         false,                           // reinterpret_input_as_3d
                                         fc_info.retain_internal_weights, // retain_internal_weights
                                         gemmlowp_output_stage,           // gemmlowp_output_stage
                                         fc_info.fp_mixed_precision,      // fp_mixed_precision
                                         true,                            // broadcast_bias
                                         fc_info.activation_info);        // activation_info

    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo input_quantization_info   = input->info()->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->info()->quantization_info();

        input->info()->set_quantization_info(QuantizationInfo(input_quantization_info.uniform().scale, -input_quantization_info.uniform().offset));
        weights->info()->set_quantization_info(QuantizationInfo(weights_quantization_info.uniform().scale, -weights_quantization_info.uniform().offset));

        // Configure gemmlowp function
        _mm_gemmlowp.configure(compile_context, input, weights, bias, output, gemm_info);

        // Revert back QuantizatioInfo as input and weights could be used in other fully connected layers
        input->info()->set_quantization_info(input_quantization_info);
        weights->info()->set_quantization_info(weights_quantization_info);
    }
    else
    {
        // Configure matrix multiply kernel
        _mm_gemm.configure(compile_context, input, weights, bias, output, 1.f, 1.f, gemm_info);
    }
}

void CLFullyConnectedLayer::configure_conv_fc(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output,
                                              const FullyConnectedLayerInfo &fc_info)
{
    ARM_COMPUTE_ERROR_ON((weights->info()->dimension(1) != (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))));

    // If the fully connected layer is called after a convolution layer, the input tensor must be linearized

    // Initialize output tensor for flatten
    TensorShape shape_flatten = compute_flatten_shape(input->info());
    _flatten_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_flatten).set_data_layout(DataLayout::NCHW));

    // Configure flatten kernel
    _memory_group.manage(&_flatten_output);
    _flatten_layer.configure(compile_context, input, &_flatten_output);

    // Configure matrix multiply kernel
    configure_mm(compile_context, &_flatten_output, weights, bias, output, fc_info);

    // Allocate the output tensor for flatten once all the configure methods have been called
    _flatten_output.allocator()->allocate();
}

void CLFullyConnectedLayer::configure_fc_fc(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output,
                                            const FullyConnectedLayerInfo &fc_info)
{
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != weights->info()->dimension(1));

    // Configure matrix multiply kernel
    configure_mm(compile_context, input, weights, bias, output, fc_info);
}

void CLFullyConnectedLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                      FullyConnectedLayerInfo fc_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, fc_info);
}

void CLFullyConnectedLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                      FullyConnectedLayerInfo fc_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(CLFullyConnectedLayer::validate(input->info(),
                                                               weights->info(),
                                                               biases != nullptr ? biases->info() : nullptr,
                                                               output->info(),
                                                               fc_info));

    _are_weights_converted = true;
    _are_weights_reshaped  = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
    _is_fc_after_conv      = true;
    _is_quantized          = is_data_type_quantized_asymmetric(input->info()->data_type());
    _is_prepared           = fc_info.retain_internal_weights;
    _original_weights      = weights;

    if(_weights_manager)
    {
        _weights_manager->manage(weights);
    }

    const ICLTensor *weights_to_use = weights;

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

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
            _reshape_weights_managed_function.configure(compile_context, weights);
            weights_to_use = utils::cast::polymorphic_downcast<ICLTensor *>(_weights_manager->acquire(weights, &_reshape_weights_managed_function));
        }
        else
        {
            // Reshape the weights
            _reshape_weights_function.configure(compile_context, weights, &_reshape_weights_output);
            weights_to_use = &_reshape_weights_output;
        }
    }

    // Convert weights if needed
    if(_is_fc_after_conv && (input->info()->data_layout() != fc_info.weights_trained_layout))
    {
        if(_weights_manager && _weights_manager->are_weights_managed(weights_to_use))
        {
            _convert_weights_managed.configure(compile_context, weights_to_use,
                                               input->info()->tensor_shape(),
                                               fc_info.weights_trained_layout);
            weights_to_use = utils::cast::polymorphic_downcast<ICLTensor *>(_weights_manager->acquire(weights, &_convert_weights_managed));
        }
        else
        {
            // Convert weights
            _convert_weights.configure(compile_context, weights_to_use,
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
        configure_conv_fc(compile_context, input, weights_to_use, biases, output, fc_info);
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        configure_fc_fc(compile_context, input, weights_to_use, biases, output, fc_info);
    }
}

Status CLFullyConnectedLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                       FullyConnectedLayerInfo fc_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(fc_info.activation_info.enabled() && is_data_type_quantized(input->data_type()) && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::RELU
                                && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::BOUNDED_RELU && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU);

    bool weights_reshaped = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
    bool is_fc_after_conv = true;

    const ITensorInfo &flatten_input     = TensorInfo(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_flatten_shape(input)).set_data_layout(DataLayout::NCHW));
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
        ARM_COMPUTE_RETURN_ON_ERROR(CLFullyConnectedLayerReshapeWeights::validate(weights, &reshaped_weights));
        weights_to_use = &reshaped_weights;
    }

    if(is_fc_after_conv && (input->data_layout() != fc_info.weights_trained_layout))
    {
        // Validate convert weights kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CLConvertFullyConnectedWeights::validate(weights_to_use,
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
        ARM_COMPUTE_RETURN_ON_ERROR(CLFlattenLayer::validate(input, &flatten_input));
        input_to_use = &flatten_input;
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != weights_to_use->dimension(1));
    }

    // Validate matrix multiply kernel
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(*input_to_use, *weights_to_use, biases, *output, fc_info));

    return Status{};
}

void CLFullyConnectedLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Linearize input if it comes from a convolutional layer
    if(_is_fc_after_conv)
    {
        _flatten_layer.run();
    }

    // Run matrix multiply
    if(_is_quantized)
    {
        _mm_gemmlowp.run();
    }
    else
    {
        _mm_gemm.run();
    }
}

void CLFullyConnectedLayer::prepare()
{
    if(!_is_prepared)
    {
        if(!_weights_manager)
        {
            ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());
        }

        auto release_unused = [](CLTensor * w)
        {
            if(!w->is_used())
            {
                CLScheduler::get().queue().finish();
                w->allocator()->free();
            }
        };

        // Pointer to current weights
        const ICLTensor *cur_weights = _original_weights;

        // Reshape of the weights if needed (happens only once)
        if(!_are_weights_reshaped)
        {
            if(_weights_manager && _weights_manager->are_weights_managed(_original_weights))
            {
                cur_weights = utils::cast::polymorphic_downcast<ICLTensor *>(_weights_manager->run(cur_weights, &_reshape_weights_managed_function));
            }
            else
            {
                // Run reshape weights kernel and mark weights as unused
                _reshape_weights_output.allocator()->allocate();
                _reshape_weights_function.run();

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
        if(!_is_quantized)
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
