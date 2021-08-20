/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/runtime/cpu/operators/CpuFullyConnected.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/cpu/kernels/CpuTransposeKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/cpu/operators/CpuConvertFullyConnectedWeights.h"
#include "src/runtime/cpu/operators/CpuFlatten.h"
#include "src/runtime/cpu/operators/CpuGemm.h"
#include "src/runtime/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"
#include "src/runtime/cpu/utils/CpuAuxTensorHandler.h"

namespace arm_compute
{
namespace cpu
{
using namespace arm_compute::experimental;
using namespace arm_compute::misc::shape_calculator;

namespace
{
// Get min, max bound of a quantized asymmetric dst tensor, with the effect of fused activation
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

Status get_gemmlowp_output_stage_info(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const ActivationLayerInfo &act,
                                      GEMMLowpOutputStageInfo &gemmlowp_output_stage_info)
{
    const auto                    data_type = src->data_type();
    const QuantizationInfo        oq_info   = dst->quantization_info();
    const UniformQuantizationInfo iq_unif   = src->quantization_info().uniform();
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

Status validate_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const ActivationLayerInfo &act)
{
    if(is_data_type_quantized_asymmetric(src->data_type()))
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate src and weights offset
        const QuantizationInfo src_quantization_info(src->quantization_info().uniform().scale, -src->quantization_info().uniform().offset);
        const QuantizationInfo weights_quantization_info(weights->quantization_info().uniform().scale, -weights->quantization_info().uniform().offset);

        GEMMLowpOutputStageInfo gemmlowp_output_stage_info;
        ARM_COMPUTE_RETURN_ON_ERROR(get_gemmlowp_output_stage_info(src, weights, dst, act, gemmlowp_output_stage_info));

        GEMMInfo gemm_info;
        gemm_info.set_gemmlowp_output_stage(gemmlowp_output_stage_info);

        // Validate gemmlowp function
        TensorInfo src_info     = src->clone()->set_quantization_info(src_quantization_info);
        TensorInfo weights_info = weights->clone()->set_quantization_info(weights_quantization_info);
        ARM_COMPUTE_RETURN_ON_ERROR(CpuGemmLowpMatrixMultiplyCore::validate(&src_info,
                                                                            &weights_info,
                                                                            biases,
                                                                            dst,
                                                                            gemm_info));
    }
    else
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CpuGemm::validate(src, weights, biases, dst, 1.f, 1.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run */)));
    }

    return Status{};
}
} // namespace

CpuFullyConnected::CpuFullyConnected()
    : _flatten(nullptr),
      _convert_weights(nullptr),
      _transpose_weights(nullptr),
      _mm_gemm(nullptr),
      _mm_gemmlowp(nullptr),
      _flattened_src(),
      _converted_weights(),
      _reshaped_weights(),
      _trans_weights(),
      _trans_weights_idx(AuxTensorIdx::Count),
      _aux_mem(Count),
      _needs_weights_conversion(false),
      _needs_weights_reshape(false),
      _is_fc_after_conv(false),
      _is_quantized_asymmetric(false),
      _is_prepared(false)

{
}

CpuFullyConnected::~CpuFullyConnected() = default;

void CpuFullyConnected::configure_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ActivationLayerInfo &act)
{
    if(_is_quantized_asymmetric)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate src and weights offset
        const QuantizationInfo src_quantization_info(src->quantization_info().uniform().scale, -src->quantization_info().uniform().offset);
        const QuantizationInfo weights_quantization_info(weights->quantization_info().uniform().scale, -weights->quantization_info().uniform().offset);

        TensorInfo src_info     = src->clone()->set_quantization_info(src_quantization_info);
        TensorInfo weights_info = weights->clone()->set_quantization_info(weights_quantization_info);

        // Configure gemmlowp function and output stage for asymmetric quantized types
        GEMMLowpOutputStageInfo gemmlowp_output_stage_info;
        const Status            status = get_gemmlowp_output_stage_info(&src_info, &weights_info, dst, act, gemmlowp_output_stage_info);
        ARM_COMPUTE_ERROR_ON(status.error_code() != ErrorCode::OK);

        GEMMInfo gemm_info;
        gemm_info.set_gemmlowp_output_stage(gemmlowp_output_stage_info);
        gemm_info.set_activation_info(act);
        _mm_gemmlowp = std::make_unique<CpuGemmLowpMatrixMultiplyCore>();
        _mm_gemmlowp->configure(&src_info, &weights_info, biases, dst, gemm_info);
    }
    else
    {
        // Configure matrix multiply kernel
        GEMMInfo gemm_info(false, false, true /* Reshape weights only for the first run */);
        gemm_info.set_activation_info(act);
        _mm_gemm = std::make_unique<CpuGemm>();
        _mm_gemm->configure(src, weights, biases, dst, 1.f, 1.0f, gemm_info);
    }
}

void CpuFullyConnected::configure_conv_fc(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ActivationLayerInfo &act)
{
    ARM_COMPUTE_ERROR_ON((weights->dimension(1) != (src->dimension(0) * src->dimension(1) * src->dimension(2))));

    // If the fully connected layer is called after a convolution layer, the src tensor must be linearized

    // Initialize output tensor for flatten
    auto_init_if_empty(_flattened_src, src->clone()->set_tensor_shape(compute_flatten_shape(src)));

    _flatten = std::make_unique<CpuFlatten>();
    _flatten->configure(src, &_flattened_src);

    // Configure matrix multiply kernel
    configure_mm(&_flattened_src, weights, biases, dst, act);
}

void CpuFullyConnected::configure_fc_fc(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ActivationLayerInfo &act)
{
    ARM_COMPUTE_ERROR_ON(src->dimension(0) != weights->dimension(1));

    // Configure matrix multiply kernel
    configure_mm(src, weights, biases, dst, act);
}

void CpuFullyConnected::configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst,
                                  FullyConnectedLayerInfo fc_info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(CpuFullyConnected::validate(src,
                                                           weights,
                                                           biases != nullptr ? biases : nullptr,
                                                           dst,
                                                           fc_info));

    _needs_weights_conversion = false;
    _needs_weights_reshape    = fc_info.transpose_weights ? !fc_info.are_weights_reshaped : false;
    _needs_weights_reshape    = _needs_weights_reshape && !fc_info.retain_internal_weights;
    _is_fc_after_conv         = true;
    _is_quantized_asymmetric  = is_data_type_quantized_asymmetric(src->data_type());
    _is_prepared              = false;
    _trans_weights_idx        = AuxTensorIdx::Count;

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    const ITensorInfo *weights_to_use = weights;

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

    // Reshape weights if needed
    if(_needs_weights_reshape)
    {
        // Reshape the weights
        _transpose_weights = std::make_unique<kernels::CpuTransposeKernel>();
        _transpose_weights->configure(weights, &_reshaped_weights);
        weights_to_use     = &_reshaped_weights;
        _trans_weights_idx = AuxTensorIdx::TransposedWeights;
    }

    // Convert weights if needed
    if(_is_fc_after_conv && (src->data_layout() != fc_info.weights_trained_layout))
    {
        // Convert weights
        _convert_weights = std::make_unique<CpuConvertFullyConnectedWeights>();
        _convert_weights->configure(weights_to_use,
                                    &_converted_weights,
                                    src->tensor_shape(),
                                    fc_info.weights_trained_layout);

        weights_to_use            = &_converted_weights;
        _needs_weights_conversion = true;
        _trans_weights_idx        = AuxTensorIdx::ConvertedWeights;
    }

    if(_is_fc_after_conv)
    {
        // Fully Connected layer after a Convolution Layer without batches
        configure_conv_fc(src, weights_to_use, biases, dst, fc_info.activation_info);
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        configure_fc_fc(src, weights_to_use, biases, dst, fc_info.activation_info);
    }

    // Retain the tensorinfo with the weights to use
    if(_needs_weights_reshape || _needs_weights_conversion)
    {
        _trans_weights = *weights_to_use;
    }

    // Set auxiliary memory requirements
    auto gemm_mem_req = (_is_quantized_asymmetric) ? _mm_gemmlowp->workspace() : _mm_gemm->workspace();
    for(unsigned int i = 0; i < gemm_mem_req.size(); ++i)
    {
        _aux_mem[i] = gemm_mem_req[i];
    }

    if(_aux_mem[Pretranspose].size > 0)
    {
        // Release permuted weights at the of prepare as they are further transposed by the assembly dispatch
        _aux_mem[TransposedWeights] = MemoryInfo(offset_int_vec(TransposedWeights), MemoryLifetime::Prepare, _reshaped_weights.total_size());
        _aux_mem[ConvertedWeights]  = MemoryInfo(offset_int_vec(ConvertedWeights), MemoryLifetime::Prepare, _converted_weights.total_size());
    }
    else
    {
        _aux_mem[TransposedWeights] = MemoryInfo(offset_int_vec(TransposedWeights), _needs_weights_conversion ? MemoryLifetime::Prepare : MemoryLifetime::Persistent, _reshaped_weights.total_size());
        _aux_mem[ConvertedWeights]  = MemoryInfo(offset_int_vec(ConvertedWeights), MemoryLifetime::Persistent, _converted_weights.total_size());
    }
    _aux_mem[FlattenedSrc] = MemoryInfo(offset_int_vec(FlattenedSrc), MemoryLifetime::Temporary, _flattened_src.total_size());
}

Status CpuFullyConnected::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                   FullyConnectedLayerInfo fc_info)
{
    ARM_COMPUTE_UNUSED(fc_info.retain_internal_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(biases != nullptr && biases->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(fc_info.activation_info.enabled() && is_data_type_quantized(src->data_type()) && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::RELU
                                && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::BOUNDED_RELU && fc_info.activation_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!fc_info.constant_weights, "Non-constant weights are currently not supported");

    bool weights_reshaped = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
    bool is_fc_after_conv = true;

    const ITensorInfo &flatten_src       = TensorInfo(src->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_flatten_shape(src)));
    const ITensorInfo &reshaped_weights  = TensorInfo(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_transposed_shape(*weights)));
    const ITensorInfo &converted_weights = weights_reshaped ? TensorInfo(weights->clone()->set_is_resizable(true).reset_padding()) : TensorInfo(*reshaped_weights.clone());

    // With the Fully Connected layer we can have 4 different cases:
    //  1) Convolution layer -> Fully Connected layer without batches
    //  2) Fully Connected layer -> Fully Connected layer without batches
    //  3) Convolution layer -> Fully Connected layer with batches
    //  4) Fully Connected layer -> Fully Connected layer with batches

    const ITensorInfo *src_to_use     = src;
    const ITensorInfo *weights_to_use = weights;

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
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuTransposeKernel::validate(weights, &reshaped_weights));
        weights_to_use = &reshaped_weights;
    }

    if(is_fc_after_conv && (src->data_layout() != fc_info.weights_trained_layout))
    {
        // Validate convert weights kernel
        ARM_COMPUTE_RETURN_ON_ERROR(CpuConvertFullyConnectedWeights::validate(weights_to_use,
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
        ARM_COMPUTE_RETURN_ON_ERROR(CpuFlatten::validate(src, &flatten_src));
        src_to_use = &flatten_src;
    }
    else
    {
        // Fully Connected layer after a Fully Connected Layer without batches
        ARM_COMPUTE_RETURN_ERROR_ON(src->dimension(0) != weights_to_use->dimension(1));
    }
    // Validate matrix multiply kernel
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(src_to_use, weights_to_use, biases, dst, fc_info.activation_info));

    return Status{};
}

void CpuFullyConnected::run(ITensorPack &tensors)
{
    prepare(tensors);

    auto src = tensors.get_const_tensor(ACL_SRC_0);

    CpuAuxTensorHandler flattened_src(offset_int_vec(FlattenedSrc), _flattened_src, tensors, false);
    CpuAuxTensorHandler transformed_wei(offset_int_vec(_trans_weights_idx), _trans_weights, tensors, false);

    // Linearize src if it comes from a convolutional layer
    if(_is_fc_after_conv)
    {
        ITensorPack flatten_pack{ { ACL_SRC, src }, { ACL_DST, flattened_src.get() } };
        _flatten->run(flatten_pack);
    }

    ITensorPack gemm_pack = tensors;
    gemm_pack.add_const_tensor(ACL_SRC_0, (_is_fc_after_conv) ? flattened_src.get() : src);
    if(_needs_weights_reshape || _needs_weights_conversion)
    {
        gemm_pack.add_const_tensor(ACL_SRC_1, transformed_wei.get());
    }

    // Run matrix multiply
    if(_is_quantized_asymmetric)
    {
        _mm_gemmlowp->run(gemm_pack);
    }
    else
    {
        _mm_gemm->run(gemm_pack);
    }
}

void CpuFullyConnected::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        auto weights = tensors.get_const_tensor(ACL_SRC_1);

        CpuAuxTensorHandler reshaped_weights(offset_int_vec(TransposedWeights), _reshaped_weights, tensors, false);
        CpuAuxTensorHandler converted_weights(offset_int_vec(ConvertedWeights), _converted_weights, tensors, false);

        // Pointer to current weights
        const ITensor *cur_weights = weights;

        // Reshape of the weights (happens only once)
        if(_needs_weights_reshape)
        {
            // Run reshape weights kernel and mark weights as unused
            ITensorPack transpose_pack{ { ACL_SRC, weights }, { ACL_DST, reshaped_weights.get() } };
            NEScheduler::get().schedule_op(_transpose_weights.get(), Window::DimY, _transpose_weights->window(), transpose_pack);

            cur_weights->mark_as_unused();
            cur_weights = reshaped_weights.get();
        }

        // Convert weights if needed (happens only once)
        if(_needs_weights_conversion)
        {
            ITensorPack convert_pack{ { ACL_SRC, cur_weights }, { ACL_DST, converted_weights.get() } };
            _convert_weights->run(convert_pack);

            cur_weights->mark_as_unused();
            cur_weights = converted_weights.get();
        }

        ITensorPack gemm_pack = tensors;
        gemm_pack.add_const_tensor(ACL_SRC_1, cur_weights);

        // Prepare GEMM prepare and release unused weights
        if(!_is_quantized_asymmetric)
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

experimental::MemoryRequirements CpuFullyConnected::workspace() const
{
    return _aux_mem;
}
} // namespace cpu
} // namespace arm_compute
