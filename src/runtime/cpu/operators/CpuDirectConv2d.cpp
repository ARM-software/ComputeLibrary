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
#include "src/runtime/cpu/operators/CpuDirectConv2d.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace arm_compute
{
namespace cpu
{
CpuDirectConv2d::~CpuDirectConv2d() = default;

CpuDirectConv2d::CpuDirectConv2d(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _output_stage_kernel(), _conv_kernel(), _input_border_handler(), _activationlayer_function(), _accumulator(), _has_bias(false),
      _is_activationlayer_enabled(false), _dim_split(Window::DimZ), _is_padding_required()
{
}

void CpuDirectConv2d::configure(ITensorInfo *src, ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *dst, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON(src->data_layout() == DataLayout::UNKNOWN);
    _output_stage_kernel  = std::make_unique<kernels::CpuDirectConv2dOutputStageKernel>();
    _conv_kernel          = std::make_unique<kernels::CpuDirectConv2dKernel>();
    _input_border_handler = std::make_unique<NEFillBorderKernel>();

    // Free accumulator
    if(_accumulator.buffer() != nullptr)
    {
        _accumulator.allocator()->free();
    }

    _dim_split = src->data_layout() == DataLayout::NCHW ? Window::DimZ : Window::DimY;

    // Check if bias should be added in the convolution result
    _has_bias = (bias != nullptr);

    _conv_kernel->configure(src, weights, dst, conv_info);
    if(_has_bias)
    {
        _output_stage_kernel->configure(dst, bias);
    }
    _is_padding_required = !_conv_kernel->border_size().empty();

    if(_is_padding_required)
    {
        // Add zero padding XY
        _input_border_handler->configure(src, _conv_kernel->border_size(), BorderMode::CONSTANT, PixelValue(static_cast<float>(0.f)));
    }

    //Configure Activation Layer
    _is_activationlayer_enabled = act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function = std::make_unique<CpuActivation>();
        _activationlayer_function->configure(dst, dst, act_info);
    }
}

Status CpuDirectConv2d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *dst, const PadStrideInfo &conv_info,
                                 const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);

    // output might not be initialized since it can be an intermediate tensor of another layer
    DataType   data_type = src->data_type();
    TensorInfo accumulator(dst->clone()->set_is_resizable(true).reset_padding().set_data_type(data_type));

    // Validate Convolution kernel
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuDirectConv2dKernel::validate(src, weights, &accumulator, conv_info));

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->dimension(0) != weights->dimension(3),
                                        "Biases size and number of input feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(bias->num_dimensions() > 1, "Biases should be one dimensional");
    }

    // Validate bias kernel
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuDirectConv2dOutputStageKernel::validate(&accumulator, bias, dst));

    if(act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CpuActivation::validate(dst, nullptr, act_info));
    }

    return Status{};
}

void CpuDirectConv2d::run(ITensorPack &tensors)
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    auto src  = tensors.get_tensor(TensorType::ACL_SRC_0);
    auto bias = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    if(_is_padding_required)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC_DST, src);
        NEScheduler::get().schedule_op(_input_border_handler.get(), Window::DimZ, _input_border_handler->window(), pack);
    }
    NEScheduler::get().schedule_op(_conv_kernel.get(), _dim_split, _conv_kernel->window(), tensors);
    if(_has_bias)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC_0, dst);
        pack.add_tensor(TensorType::ACL_SRC_1, bias);
        pack.add_tensor(TensorType::ACL_DST, dst);
        NEScheduler::get().schedule_op(_output_stage_kernel.get(), Window::DimY, _output_stage_kernel->window(), pack);
    }

    if(_is_activationlayer_enabled)
    {
        ITensorPack pack;
        pack.add_tensor(TensorType::ACL_SRC, dst);
        pack.add_tensor(TensorType::ACL_DST, dst);
        _activationlayer_function->run(pack);
    }
}
} // namespace cpu
} // namespace arm_compute
