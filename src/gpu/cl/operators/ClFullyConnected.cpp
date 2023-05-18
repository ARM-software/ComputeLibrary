/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "src/gpu/cl/operators/ClFullyConnected.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/operators/ClConvertFullyConnectedWeights.h"
#include "src/gpu/cl/operators/ClFlatten.h"
#include "src/gpu/cl/operators/ClGemm.h"
#include "src/gpu/cl/operators/ClGemmLowpMatrixMultiplyCore.h"
#include "src/gpu/cl/operators/ClTranspose.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

#include "src/common/utils/Log.h"
#include "support/Cast.h"

#include <algorithm>

namespace arm_compute
{
namespace opencl
{
using namespace arm_compute::experimental;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status construct_gemmlowp_output_stage(const ITensorInfo &src, const ITensorInfo &weights, const ITensorInfo &dst,
                                       GEMMLowpOutputStageInfo &gemmlowp_output_stage, ActivationLayerInfo activation_info)
{
    gemmlowp_output_stage.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    gemmlowp_output_stage.gemmlowp_offset     = 0;
    gemmlowp_output_stage.gemmlowp_multiplier = 0;
    gemmlowp_output_stage.gemmlowp_shift      = 0;

    const auto data_type = src.data_type();

    // Configure output stage for quantized case
    if(is_data_type_quantized_asymmetric(data_type))
    {
        const QuantizationInfo        oq_info = dst.quantization_info();
        const UniformQuantizationInfo iq_unif = src.quantization_info().uniform();
        const UniformQuantizationInfo wq_unif = weights.quantization_info().uniform();
        const UniformQuantizationInfo oq_unif = oq_info.uniform();

        const auto output_quant_info = (dst.total_size() == 0) ? iq_unif : oq_unif;

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

Status validate_mm(const ITensorInfo &src, const ITensorInfo &weights, const ITensorInfo *bias, const ITensorInfo &dst, const FullyConnectedLayerInfo &fc_info)
{
    GEMMLowpOutputStageInfo gemmlowp_output_stage;
    ARM_COMPUTE_RETURN_ON_ERROR(construct_gemmlowp_output_stage(src, weights, dst, gemmlowp_output_stage, fc_info.activation_info));

    const GEMMInfo &gemm_info = GEMMInfo(false,                           // is_a_reshaped
                                         false,                           // is_b_reshaped
                                         true,                            // reshape_b_only_on_first_run
                                         0,                               // depth_output_gemm3d
                                         false,                           // reinterpret_input_as_3d
                                         fc_info.retain_internal_weights, // retain_internal_weights
                                         gemmlowp_output_stage,           // gemmlowp_output_stage
                                         fc_info.fp_mixed_precision,      // fp_mixed_precision
                                         false,                           // fast_math
                                         true,                            // broadcast_bias
                                         ActivationLayerInfo());          // activation_info

    if(is_data_type_quantized_asymmetric(src.data_type()))
    {
        const UniformQuantizationInfo iq_info = src.quantization_info().uniform();
        const UniformQuantizationInfo wq_info = weights.quantization_info().uniform();

        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate src and weights offset
        const QuantizationInfo src_quantization_info(iq_info.scale, -iq_info.offset);
        const QuantizationInfo weights_quantization_info(wq_info.scale, -wq_info.offset);

        // Validate gemmlowp function
        ARM_COMPUTE_RETURN_ON_ERROR(ClGemmLowpMatrixMultiplyCore::validate(&src.clone()->set_quantization_info(src_quantization_info),
                                                                           &weights.clone()->set_quantization_info(weights_quantization_info),
                                                                           bias,
                                                                           &dst,
                                                                           gemm_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(ClGemm::validate(&src, &weights, bias, &dst, 1.f, 1.f, gemm_info));
    }

    return Status{};
}
} // namespace

ClFullyConnected::ClFullyConnected()
    : _convert_weights(nullptr),
      _flatten(nullptr),
      _reshape_weights(nullptr),
      _mm_gemm(nullptr),
      _mm_gemmlowp(nullptr),
      _aux_mem(Count)
{
}

ClFullyConnected::~ClFullyConnected() = default;

void ClFullyConnected::configure_mm(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *dst,
                                    const FullyConnectedLayerInfo &fc_info)
{
    GEMMLowpOutputStageInfo gemmlowp_output_stage;
    construct_gemmlowp_output_stage(*src, *weights, *dst, gemmlowp_output_stage, fc_info.activation_info);

    const GEMMInfo &gemm_info = GEMMInfo(false,                           // is_a_reshaped
                                         false,                           // is_b_reshaped
                                         !_dynamic_weights,               // reshape_b_only_on_first_run
                                         0,                               // depth_output_gemm3d
                                         false,                           // reinterpret_input_as_3d
                                         fc_info.retain_internal_weights, // retain_internal_weights
                                         gemmlowp_output_stage,           // gemmlowp_output_stage
                                         fc_info.fp_mixed_precision,      // fp_mixed_precision
                                         false,                           // fast_math
                                         true,                            // broadcast_bias
                                         fc_info.activation_info);        // activation_info

    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo src_quantization_info     = src->quantization_info();
        const QuantizationInfo weights_quantization_info = weights->quantization_info();

        TensorInfo src_info     = src->clone()->set_quantization_info(src_quantization_info);
        TensorInfo weights_info = weights->clone()->set_quantization_info(weights_quantization_info);

        src_info.set_quantization_info(QuantizationInfo(src_quantization_info.uniform().scale, -src_quantization_info.uniform().offset));
        weights_info.set_quantization_info(QuantizationInfo(weights_quantization_info.uniform().scale, -weights_quantization_info.uniform().offset));

        // Configure gemmlowp function
        _mm_gemmlowp = std::make_unique<ClGemmLowpMatrixMultiplyCore>();
        _mm_gemmlowp->configure(compile_context, &src_info, &weights_info, bias, dst, gemm_info);
    }
    else
    {
        // Configure matrix multiply kernel
        _mm_gemm = std::make_unique<ClGemm>();
        _mm_gemm->configure(compile_context, src, weights, bias, dst, 1.f, 1.f, gemm_info);
    }
}

void ClFullyConnected::configure_conv_fc(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *dst,
                                         const FullyConnectedLayerInfo &fc_info)
{
    ARM_COMPUTE_ERROR_ON((weights->dimension(1) != (src->dimension(0) * src->dimension(1) * src->dimension(2))));

    // If the fully connected layer is called after a convolution layer, the input tensor must be linearized

    // Initialize output tensor for flatten
    _flattened_src = src->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_flatten_shape(src)).set_data_layout(DataLayout::NCHW);

    // Configure flatten kernel
    _flatten = std::make_unique<ClFlatten>();
    _flatten->configure(compile_context, src, &_flattened_src);

    // Configure matrix multiply kernel
    configure_mm(compile_context, &_flattened_src, weights, bias, dst, fc_info);
}

void ClFullyConnected::configure_fc_fc(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *dst,
                                       const FullyConnectedLayerInfo &fc_info)
{
    ARM_COMPUTE_ERROR_ON(src->dimension(0) != weights->dimension(1));

    // Configure matrix multiply kernel
    configure_mm(compile_context, src, weights, bias, dst, fc_info);
}

void ClFullyConnected::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                                 FullyConnectedLayerInfo fc_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);

    // Perform validate step
    ARM_COMPUTE_ERROR_THROW_ON(ClFullyConnected::validate(src, weights, biases, dst, fc_info));
    ARM_COMPUTE_LOG_PARAMS(src, weights, biases, dst, fc_info);

    _are_weights_converted = true;
    _are_weights_reshaped  = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
    _is_fc_after_conv      = true;
    _is_quantized          = is_data_type_quantized_asymmetric(src->data_type());
    _is_prepared           = fc_info.retain_internal_weights;
    _weights_to_use        = TensorInfo(*weights);
    _weights_to_use_idx    = ACL_SRC_1;
    _dynamic_weights       = !weights->are_values_constant() && !_are_weights_reshaped;

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    // Check if we have a fully connected layer with batches
    const bool is_batched_fc_layer = dst->dimension(1) > 1;
    if(is_batched_fc_layer)
    {
        _is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) && (std::equal(src->tensor_shape().cbegin() + 3,
                                                                                  src->tensor_shape().cend(),
                                                                                  dst->tensor_shape().cbegin() + 1));
    }
    else
    {
        _is_fc_after_conv = src->num_dimensions() > 1;
    }

    ITensorInfo *weights_used = weights;

    // Reshape weights if needed
    if(!_are_weights_reshaped)
    {
        // Reshape the weights
        _reshape_weights = std::make_unique<ClTranspose>();
        _reshape_weights->configure(compile_context, weights, &_reshaped_weights);
        weights_used        = &_reshaped_weights;
        _weights_to_use_idx = offset_int_vec(TransposedWeights);
    }

    // Convert weights if needed
    if(_is_fc_after_conv && (src->data_layout() != fc_info.weights_trained_layout))
    {
        // Convert weights
        _convert_weights = std::make_unique<ClConvertFullyConnectedWeights>();
        _convert_weights->configure(compile_context,
                                    weights_used,
                                    &_converted_weights,
                                    src->tensor_shape(),
                                    fc_info.weights_trained_layout);

        weights_used           = &_converted_weights;
        _weights_to_use_idx    = offset_int_vec(ConvertedWeights);
        _are_weights_converted = false;
    }

    if(_is_fc_after_conv)
    {
        // Fully Connected layer after a Convolution Layer without batches
        configure_conv_fc(compile_context, src, weights_used, biases, dst, fc_info);
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        configure_fc_fc(compile_context, src, weights_used, biases, dst, fc_info);
    }
    // Update TensorInfo of final weights used (Need to be done in the end due to padding expansion)
    _weights_to_use = *weights_used;

    // Set auxiliary memory requirements
    auto gemm_mem_req = (_is_quantized) ? _mm_gemmlowp->workspace() : _mm_gemm->workspace();
    for(unsigned int i = 0; i < gemm_mem_req.size(); ++i)
    {
        _aux_mem[i] = gemm_mem_req[i];
    }
    if(_aux_mem[1].size > 0 || _aux_mem[2].size > 0) // Persistent weights memory on GEMMs
    {
        // Release permuted weights at the of prepare as they are further transposed by the assembly dispatch
        // Keep all the auxiliary tensors in case of dynamic weights as they are recalculated every time
        _aux_mem[TransposedWeights] = MemoryInfo(
            offset_int_vec(TransposedWeights),
            _dynamic_weights ? MemoryLifetime::Temporary : MemoryLifetime::Prepare,
            _reshaped_weights.total_size());
        _aux_mem[ConvertedWeights]  = MemoryInfo(
            offset_int_vec(ConvertedWeights),
            _dynamic_weights ? MemoryLifetime::Temporary : MemoryLifetime::Prepare,
            _converted_weights.total_size());
    }
    else
    {
        // Release permuted weights at the of prepare as they are further transposed by the assembly dispatch
        const auto transposed_wei_lft = (_weights_to_use_idx == offset_int_vec(TransposedWeights)) ? MemoryLifetime::Persistent : MemoryLifetime::Prepare;
        const auto converted_wei_lft  = (_weights_to_use_idx == offset_int_vec(ConvertedWeights)) ? MemoryLifetime::Persistent : MemoryLifetime::Prepare;

        _aux_mem[TransposedWeights] = MemoryInfo(
            offset_int_vec(TransposedWeights),
            _dynamic_weights ? MemoryLifetime::Temporary : transposed_wei_lft,
            _reshaped_weights.total_size());
        _aux_mem[ConvertedWeights] = MemoryInfo(
            offset_int_vec(ConvertedWeights),
            _dynamic_weights ? MemoryLifetime::Temporary : converted_wei_lft,
            _converted_weights.total_size());
    }
    _aux_mem[FlattenedSrc] = MemoryInfo(offset_int_vec(FlattenedSrc), MemoryLifetime::Temporary, _flattened_src.total_size());
}

Status ClFullyConnected::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                  FullyConnectedLayerInfo fc_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(fc_info.activation_info.enabled() && is_data_type_quantized(src->data_type()) && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::RELU
                                && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::BOUNDED_RELU && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU);

    bool weights_reshaped = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
    bool is_fc_after_conv = true;

    const ITensorInfo &flatten_src       = TensorInfo(src->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_flatten_shape(src)).set_data_layout(DataLayout::NCHW));
    const ITensorInfo &reshaped_weights  = TensorInfo(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_transposed_shape(*weights)));
    const ITensorInfo &converted_weights = weights_reshaped ? TensorInfo(weights->clone()->set_is_resizable(true).reset_padding()) : TensorInfo(*reshaped_weights.clone());

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    const ITensorInfo *src_to_use     = src;
    const ITensorInfo *weights_to_use = weights;

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
        if(is_data_type_quantized(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
        }
    }

    // Check if we have a fully connected layer with batches
    const bool is_batched_fc_layer = dst->dimension(1) > 1;
    if(is_batched_fc_layer)
    {
        is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) && (std::equal(src->tensor_shape().cbegin() + 3,
                                                                                 src->tensor_shape().cend(),
                                                                                 dst->tensor_shape().cbegin() + 1));
    }
    else
    {
        is_fc_after_conv = src->num_dimensions() > 1;
    }

    if(!weights_reshaped)
    {
        // Validate reshape weights kernel
        ARM_COMPUTE_RETURN_ON_ERROR(ClTranspose::validate(weights, &reshaped_weights));
        weights_to_use = &reshaped_weights;
    }

    if(is_fc_after_conv && (src->data_layout() != fc_info.weights_trained_layout))
    {
        // Validate convert weights kernel
        ARM_COMPUTE_RETURN_ON_ERROR(ClConvertFullyConnectedWeights::validate(weights_to_use,
                                                                             &converted_weights,
                                                                             src->tensor_shape(),
                                                                             fc_info.weights_trained_layout));
        weights_to_use = &converted_weights;
    }

    if(is_fc_after_conv)
    {
        // Fully Connected layer after a Convolution Layer without batches
        ARM_COMPUTE_RETURN_ERROR_ON((weights_to_use->dimension(1) != (src->dimension(0) * src->dimension(1) * src->dimension(2))));

        // Validate flatten kernel
        ARM_COMPUTE_RETURN_ON_ERROR(ClFlatten::validate(src, &flatten_src));
        src_to_use = &flatten_src;
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(0) != weights_to_use->dimension(1));
    }

    // Validate matrix multiply kernel
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(*src_to_use, *weights_to_use, biases, *dst, fc_info));

    return Status{};
}

void ClFullyConnected::run(ITensorPack &tensors)
{
    prepare(tensors);

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
    ++_asrt_run_count;
    ARM_COMPUTE_ERROR_ON(_dynamic_weights && _asrt_prepare_count != _asrt_run_count);
#endif // ARM_COMPUTE_ASSERTS_ENABLED

    auto src = tensors.get_const_tensor(ACL_SRC_0);

    CLAuxTensorHandler flattened_src(offset_int_vec(FlattenedSrc), _flattened_src, tensors, false);
    CLAuxTensorHandler weights(_weights_to_use_idx, _weights_to_use, tensors, false);

    // Linearize input if it comes from a convolutional layer
    if(_is_fc_after_conv)
    {
        ITensorPack flatten_pack{ { ACL_SRC, src }, { ACL_DST, flattened_src.get() } };
        _flatten->run(flatten_pack);
    }

    ITensorPack gemm_pack = tensors;
    gemm_pack.add_const_tensor(ACL_SRC_0, (_is_fc_after_conv) ? flattened_src.get() : src);
    if(_weights_to_use_idx != ACL_SRC_1)
    {
        gemm_pack.add_const_tensor(ACL_SRC_1, weights.get());
    }

    // Run matrix multiply
    if(_is_quantized)
    {
        _mm_gemmlowp->run(gemm_pack);
    }
    else
    {
        _mm_gemm->run(gemm_pack);
    }
}

void ClFullyConnected::prepare(ITensorPack &tensors)
{
    if(!_is_prepared || _dynamic_weights)
    {
#ifdef ARM_COMPUTE_ASSERTS_ENABLED
        ++_asrt_prepare_count;
        ARM_COMPUTE_ERROR_ON(!_dynamic_weights && _asrt_prepare_count > 1);
#endif // ARM_COMPUTE_ASSERTS_ENABLED

        auto weights = tensors.get_const_tensor(ACL_SRC_1);

        CLAuxTensorHandler reshaped_weights(offset_int_vec(TransposedWeights), _reshaped_weights, tensors, false);
        CLAuxTensorHandler converted_weights(offset_int_vec(ConvertedWeights), _converted_weights, tensors, false);

        // Pointer to current weights
        const ITensor *cur_weights = weights;

        // Reshape of the weights if needed
        if(!_are_weights_reshaped)
        {
            // Run reshape weights kernel and mark weights as unused
            ITensorPack transpose_pack{ { ACL_SRC, weights }, { ACL_DST, reshaped_weights.get() } };
            _reshape_weights->run(transpose_pack);

            cur_weights->mark_as_unused();
            cur_weights = reshaped_weights.get();
        }

        // Convert weights if needed
        if(!_are_weights_converted)
        {
            ITensorPack convert_pack{ { ACL_SRC, cur_weights }, { ACL_DST, converted_weights.get() } };
            _convert_weights->run(convert_pack);

            cur_weights->mark_as_unused();
            cur_weights = converted_weights.get();
        }

        ITensorPack gemm_pack = tensors;
        gemm_pack.add_const_tensor(ACL_SRC_1, cur_weights);

        // Prepare GEMM prepare and release unused weights
        if(!_is_quantized)
        {
            _mm_gemm->prepare(gemm_pack);
        }
        else
        {
            _mm_gemmlowp->prepare(gemm_pack);
        }
        _is_prepared = true;
    }
}

experimental::MemoryRequirements ClFullyConnected::workspace() const
{
    return _aux_mem;
}
} // namespace opencl
} // namespace arm_compute
