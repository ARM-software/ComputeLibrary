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
#include "src/cpu/operators/CpuDirectConv3d.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace cpu
{
CpuDirectConv3d::~CpuDirectConv3d() = default;

CpuDirectConv3d::CpuDirectConv3d(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _conv_kernel(), _activationlayer_function(), _accumulator(), _is_activationlayer_enabled(false), _dim_split(Window::DimZ)
{
}

void CpuDirectConv3d::configure(ITensorInfo *src, ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const Conv3dInfo conv_info)
{
    ARM_COMPUTE_LOG_PARAMS(src, weights, biases, dst, conv_info);
    ARM_COMPUTE_ERROR_ON(src->data_layout() != DataLayout::NDHWC);

    _conv_kernel = std::make_unique<kernels::CpuDirectConv3dKernel>();

    // Free accumulator
    if(_accumulator.buffer() != nullptr)
    {
        _accumulator.allocator()->free();
    }

    _dim_split = Window::DimY;

    _conv_kernel->configure(src, weights, biases, dst, conv_info);

    //Configure Activation Layer
    _is_activationlayer_enabled = conv_info.act_info.enabled();
    if(_is_activationlayer_enabled)
    {
        _activationlayer_function = std::make_unique<CpuActivation>();
        _activationlayer_function->configure(dst, dst, conv_info.act_info);
    }
}

Status CpuDirectConv3d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv3dInfo conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);

    // output might not be initialized since it can be an intermediate tensor of another layer
    DataType   data_type = src->data_type();
    TensorInfo accumulator(dst->clone()->set_is_resizable(true).reset_padding().set_data_type(data_type));

    // Validate Convolution kernel
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuDirectConv3dKernel::validate(src, weights, biases, &accumulator, conv_info));

    if(conv_info.act_info.enabled())
    {
        ARM_COMPUTE_RETURN_ON_ERROR(CpuActivation::validate(dst, nullptr, conv_info.act_info));
    }

    return Status{};
}

void CpuDirectConv3d::run(ITensorPack &tensors)
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    NEScheduler::get().schedule_op(_conv_kernel.get(), _dim_split, _conv_kernel->window(), tensors);

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