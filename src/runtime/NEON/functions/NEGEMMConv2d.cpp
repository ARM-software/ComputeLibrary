/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMMConv2d.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/runtime/NEON/functions/NEGEMMAssemblyDispatch.h"

#include <set>

namespace arm_compute
{
namespace
{
GEMMLowpOutputStageInfo calculate_output_stage_metadata(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const ActivationLayerInfo &act)
{
    // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
    // Extract and negate input and weights offset
    const QuantizationInfo        iqinfo    = input->quantization_info();
    const QuantizationInfo        wqinfo    = weights->quantization_info();
    const QuantizationInfo        oqinfo    = (output->total_size() == 0) ? iqinfo : output->quantization_info();
    const UniformQuantizationInfo uoqinfo   = oqinfo.uniform();
    const DataType                data_type = input->data_type();
    // Merge activation with output stage
    const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                               ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                               ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                             };
    PixelValue type_min{};
    PixelValue type_max{};
    std::tie(type_min, type_max) = get_min_max(data_type);
    int32_t min_activation = type_min.get<int32_t>();
    int32_t max_activation = type_max.get<int32_t>();
    if(supported_acts.count(act.activation()) != 0)
    {
        std::tie(min_activation, max_activation) = get_quantized_activation_min_max(act, data_type, uoqinfo);
    }
    GEMMLowpOutputStageInfo os_info;
    os_info.type                     = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    os_info.gemmlowp_offset          = uoqinfo.offset;
    os_info.gemmlowp_min_bound       = min_activation;
    os_info.gemmlowp_max_bound       = max_activation;
    os_info.is_quantized_per_channel = (weights->data_type() == DataType::QSYMM8_PER_CHANNEL);
    quantization::calculate_quantized_multipliers(iqinfo, wqinfo, oqinfo, os_info);
    return os_info;
}
AsmGemmInfo init_assembly_metadata(const Conv2dInfo &info, bool is_indirect)
{
    AsmGemmInfo asm_info;
    asm_info.method                  = is_indirect ? AsmConvMethod::Indirect : AsmConvMethod::Conv;
    asm_info.ps_info                 = info.conv_info;
    asm_info.activation_info         = info.act_info;
    asm_info.depth_output_gemm3d     = true;
    asm_info.reinterpret_input_as_3d = true;
    asm_info.padding_top             = info.conv_info.pad_top();
    asm_info.padding_left            = info.conv_info.pad_left();
    asm_info.padding_value           = 0.f;
    asm_info.negated_offsets         = false;
    return asm_info;
}
} // namespace

NEGEMMConv2d::NEGEMMConv2d(const std::shared_ptr<IMemoryManager> &memory_manager)
    : _gemm_asm_func(std::make_unique<NEGEMMAssemblyDispatch>(memory_manager)), _activation_func(), _weights_permute_func(), _original_weights(nullptr), _permuted_weights(), _is_prepared(false),
      _run_activation(false)
{
}

NEGEMMConv2d::~NEGEMMConv2d() = default;

void NEGEMMConv2d::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const Conv2dInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMConv2d::validate(input->info(),
                                                      weights->info(),
                                                      biases != nullptr ? biases->info() : nullptr,
                                                      output->info(),
                                                      info));
    _original_weights = weights;
    _weights_permute_func.configure(weights, &_permuted_weights, PermutationVector{ 3, 0, 1, 2 });

    // Configure assembly dispatch
    AsmGemmInfo asm_info = init_assembly_metadata(info, false);
    if(is_data_type_quantized(input->info()->data_type()))
    {
        asm_info.output_stage = calculate_output_stage_metadata(input->info(), weights->info(), output->info(), info.act_info);
    }
    _gemm_asm_func->configure(input, &_permuted_weights, biases, output, asm_info);

    // Configure activation
    if(info.act_info.enabled() && !_gemm_asm_func->is_activation_supported(info.act_info))
    {
        _activation_func.configure(output, nullptr, info.act_info);
        _run_activation = true;
    }
}
Status NEGEMMConv2d::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const Conv2dInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL, DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.num_groups > 1, "Grouping (num_groups != 1) is not supported on Neon");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_layout() != DataLayout::NHWC, "Data layout supported is NHWC");
    const DataType    data_type = input->data_type();
    const TensorShape i_shape   = input->tensor_shape();
    const TensorShape w_shape   = weights->tensor_shape();
    ARM_COMPUTE_RETURN_ERROR_ON(w_shape[0] != i_shape[0]);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    // Validate biases
    if(biases != nullptr)
    {
        if(is_data_type_quantized_asymmetric(data_type))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else if(data_type == DataType::BFLOAT16)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::F32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(3));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    AsmGemmInfo asm_info = init_assembly_metadata(info, false);
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMAssemblyDispatch::validate(input, weights, biases, output, asm_info));
    return Status{};
}
void NEGEMMConv2d::run()
{
    prepare();

    _gemm_asm_func->run();
    if(_run_activation)
    {
        _activation_func.run();
    }
}
void NEGEMMConv2d::prepare()
{
    if(!_is_prepared)
    {
        _permuted_weights.allocator()->allocate();
        _weights_permute_func.run();
        _original_weights->mark_as_unused();
        _is_prepared = true;
    }
}
} // namespace arm_compute
