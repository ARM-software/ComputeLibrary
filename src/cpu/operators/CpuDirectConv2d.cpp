/*
 * Copyright (c) 2021, 2024-2025 Arm Limited.
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
#include "src/cpu/operators/CpuDirectConv2d.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

namespace arm_compute
{
namespace cpu
{
CpuDirectConv2d::~CpuDirectConv2d() = default;

CpuDirectConv2d::CpuDirectConv2d(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)),
      _output_stage_kernel(),
      _conv_kernel(),
      _input_border_handler(),
      _activationlayer_function(),
      _accumulator(),
      _has_bias(false),
      _is_activationlayer_enabled(false),
      _is_padding_required()
{
}

void CpuDirectConv2d::configure(ITensorInfo               *src,
                                ITensorInfo               *weights,
                                const ITensorInfo         *bias,
                                ITensorInfo               *dst,
                                const PadStrideInfo       &conv_info,
                                const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON(src->data_layout() != DataLayout::NCHW && src->data_layout() != DataLayout::NHWC);
    ARM_COMPUTE_LOG_PARAMS(src, weights, bias, dst, conv_info, act_info);

    _output_stage_kernel  = std::make_unique<kernels::CpuDirectConv2dOutputStageKernel>();
    _conv_kernel          = std::make_unique<kernels::CpuDirectConv2dKernel>();
    _input_border_handler = std::make_unique<NEFillBorderKernel>();
    _is_nchw              = src->data_layout() == DataLayout::NCHW;
    _has_bias             = bias != nullptr;
    _is_padding_required  = !_conv_kernel->border_size().empty();

    // Free accumulator
    if (_accumulator.buffer() != nullptr)
    {
        _accumulator.allocator()->free();
    }

    ITensorInfo *input_to_use   = src;
    ITensorInfo *weights_to_use = weights;
    ITensorInfo *output_to_use  = dst;

    if (_is_nchw)
    {
        _permute_input   = std::make_unique<cpu::CpuPermute>();
        _permute_weights = std::make_unique<cpu::CpuPermute>();

        _permute_input->configure(src, &_src_perm_info, PermutationVector(2U, 0U, 1U));
        _src_perm_info.set_data_layout(DataLayout::NHWC);
        input_to_use = &_src_perm_info;

        _aux_mem[PermInput] = experimental::MemoryInfo(
            offset_int_vec(PermInput), experimental::MemoryLifetime::Temporary, input_to_use->total_size());

        _permute_weights->configure(weights, &_wei_perm_info, PermutationVector(2U, 0U, 1U));
        _wei_perm_info.set_data_layout(DataLayout::NHWC);
        weights_to_use = &_wei_perm_info;

        // @note: possible optimization to do weight transform once if the weight is constant. But, it requires changes to the API.
        _aux_mem[PermWeights] = experimental::MemoryInfo(
            offset_int_vec(PermWeights), experimental::MemoryLifetime::Temporary, weights_to_use->total_size());

        _dst_perm_info.set_data_layout(DataLayout::NHWC);
        output_to_use = &_dst_perm_info;
    }

    _conv_kernel->configure(input_to_use, weights_to_use, output_to_use, conv_info);

    if (_is_padding_required)
    {
        // Add zero padding XY
        _input_border_handler->configure(input_to_use, _conv_kernel->border_size(), BorderMode::CONSTANT,
                                         PixelValue(static_cast<float>(0.f)));
    }

    if (_is_nchw)
    {
        _permute_output = std::make_unique<cpu::CpuPermute>();
        _permute_output->configure(&_dst_perm_info, dst, PermutationVector(1U, 2U, 0U));
        _dst_perm_info.set_data_layout(DataLayout::NHWC);
        dst->set_data_layout(DataLayout::NCHW);

        _aux_mem[PermOutput] = experimental::MemoryInfo(
            offset_int_vec(PermOutput), experimental::MemoryLifetime::Temporary, _dst_perm_info.total_size());
    }

    if (_has_bias)
    {
        _output_stage_kernel->configure(dst, bias);
    }

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();
    if (_is_activationlayer_enabled)
    {
        _activationlayer_function = std::make_unique<CpuActivation>();
        _activationlayer_function->configure(dst, dst, act_info);
    }
}

Status CpuDirectConv2d::validate(const ITensorInfo         *src,
                                 const ITensorInfo         *weights,
                                 const ITensorInfo         *bias,
                                 const ITensorInfo         *dst,
                                 const PadStrideInfo       &conv_info,
                                 const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    TensorInfo acc_to_use{};
    if (src->data_layout() == DataLayout::NCHW)
    {
        TensorShape permuted_input_shape   = src->tensor_shape();
        TensorShape permuted_weights_shape = weights->tensor_shape();
        TensorShape permuted_output_shape  = dst->tensor_shape();
        permute(permuted_input_shape, PermutationVector(2U, 0U, 1U));
        permute(permuted_weights_shape, PermutationVector(2U, 0U, 1U));
        permute(permuted_output_shape, PermutationVector(2U, 0U, 1U));

        const TensorInfo permuted_input   = TensorInfo(src->clone()
                                                           ->set_is_resizable(true)
                                                           .reset_padding()
                                                           .set_tensor_shape(permuted_input_shape)
                                                           .set_data_layout(DataLayout::NHWC));
        const TensorInfo permuted_weights = TensorInfo(weights->clone()
                                                           ->set_is_resizable(true)
                                                           .reset_padding()
                                                           .set_tensor_shape(permuted_weights_shape)
                                                           .set_data_layout(DataLayout::NHWC));
        const TensorInfo permuted_output  = TensorInfo(dst->clone()
                                                           ->set_is_resizable(true)
                                                           .reset_padding()
                                                           .set_tensor_shape(permuted_output_shape)
                                                           .set_data_layout(DataLayout::NHWC));

        ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(src, &permuted_input, PermutationVector(2U, 0U, 1U)));
        ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(weights, &permuted_weights, PermutationVector(2U, 0U, 1U)));
        ARM_COMPUTE_RETURN_ON_ERROR(CpuPermute::validate(&permuted_output, dst, PermutationVector(1U, 2U, 0U)));

        // output might not be initialized since it can be an intermediate tensor of another layer
        const DataType   data_type = src->data_type();
        const TensorInfo accumulator(
            permuted_output.clone()->set_is_resizable(true).reset_padding().set_data_type(data_type));
        acc_to_use = accumulator;
        ARM_COMPUTE_RETURN_ON_ERROR(
            kernels::CpuDirectConv2dKernel::validate(&permuted_input, &permuted_weights, &accumulator, conv_info));
    }
    else
    {
        // output might not be initialized since it can be an intermediate tensor of another layer
        const DataType   data_type = src->data_type();
        const TensorInfo accumulator(dst->clone()->set_is_resizable(true).reset_padding().set_data_type(data_type));
        acc_to_use = accumulator;
        // Validate Convolution kernel
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuDirectConv2dKernel::validate(src, weights, &accumulator, conv_info));
    }

    if (bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->dimension(0) != weights->dimension(3),
                                        "Biases size and number of input feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->num_dimensions() > 1, "Biases should be one dimensional");
    }

    // Validate bias kernel
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuDirectConv2dOutputStageKernel::validate(&acc_to_use, bias, dst));

    if (act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CpuActivation::validate(dst, nullptr, act_info));
    }

    return Status{};
}

experimental::MemoryRequirements CpuDirectConv2d::workspace() const
{
    return _aux_mem;
}

void CpuDirectConv2d::run(ITensorPack &tensors)
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    auto src     = tensors.get_tensor(TensorType::ACL_SRC_0);
    auto weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto bias    = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto dst     = tensors.get_tensor(TensorType::ACL_DST);
    if (_is_nchw)
    {
        // Initialise object to handle stored permuted tensors in auxillary memory
        CpuAuxTensorHandler src_perm_handle(offset_int_vec(PermInput), _src_perm_info, tensors);
        ITensor            *src_perm = src_perm_handle.get();

        CpuAuxTensorHandler wei_perm_handle(offset_int_vec(PermWeights), _wei_perm_info, tensors);
        ITensor            *weights_perm = wei_perm_handle.get();

        CpuAuxTensorHandler dst_perm_handle(offset_int_vec(PermOutput), _dst_perm_info, tensors);
        ITensor            *dst_perm = dst_perm_handle.get();

        ITensorPack pack_perm_src;
        pack_perm_src.add_tensor(TensorType::ACL_SRC, src);
        pack_perm_src.add_tensor(TensorType::ACL_DST, src_perm);
        _permute_input->run(pack_perm_src);

        ITensorPack pack_perm_weights;
        pack_perm_weights.add_tensor(TensorType::ACL_SRC, weights);
        pack_perm_weights.add_tensor(TensorType::ACL_DST, weights_perm);
        _permute_weights->run(pack_perm_weights);

        if (_is_padding_required)
        {
            ITensorPack pack;
            pack.add_tensor(TensorType::ACL_SRC_DST, src_perm);
            NEScheduler::get().schedule_op(_input_border_handler.get(), Window::DimZ, _input_border_handler->window(),
                                           pack);
        }
        ITensorPack pack_dconv;
        pack_dconv.add_const_tensor(TensorType::ACL_SRC_0, src_perm);
        pack_dconv.add_const_tensor(TensorType::ACL_SRC_1, weights_perm);
        pack_dconv.add_tensor(TensorType::ACL_DST, dst_perm);
        NEScheduler::get().schedule_op(_conv_kernel.get(), Window::DimY, _conv_kernel->window(), pack_dconv);

        ITensorPack pack_perm_dst;
        pack_perm_dst.add_tensor(TensorType::ACL_SRC, dst_perm);
        pack_perm_dst.add_tensor(TensorType::ACL_DST, dst);
        _permute_output->run(pack_perm_dst);
    }
    else
    {
        if (_is_padding_required)
        {
            ITensorPack pack;
            pack.add_tensor(TensorType::ACL_SRC_DST, src);
            NEScheduler::get().schedule_op(_input_border_handler.get(), Window::DimZ, _input_border_handler->window(),
                                           pack);
        }
        NEScheduler::get().schedule_op(_conv_kernel.get(), Window::DimY, _conv_kernel->window(), tensors);
    }

    if (_has_bias)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC_0, dst);
        pack.add_tensor(TensorType::ACL_SRC_1, bias);
        pack.add_tensor(TensorType::ACL_DST, dst);
        NEScheduler::get().schedule_op(_output_stage_kernel.get(), Window::DimY, _output_stage_kernel->window(), pack);
    }

    if (_is_activationlayer_enabled)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC, dst);
        pack.add_tensor(TensorType::ACL_DST, dst);
        _activationlayer_function->run(pack);
    }
}
} // namespace cpu
} // namespace arm_compute
