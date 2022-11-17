/*
 * Copyright (c) 2022 Arm Limited.
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
#include "src/gpu/cl/operators/ClIndirectConv2d.h"

#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/gpu/cl/kernels/ClIndirectConv2dAddressPrecalculationKernel.h"
#include "src/gpu/cl/kernels/ClIndirectConv2dKernel.h"
#include "src/gpu/cl/kernels/direct_conv/ClDirectConvKernelConfig.h"
#include "src/gpu/cl/kernels/direct_conv/IClDirectConvKernelConfig.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

#include "src/common/utils/Log.h"

using namespace arm_compute::cl_direct_conv;

namespace arm_compute
{
namespace opencl
{
using namespace arm_compute::experimental;

namespace
{
DirectConvComputeKernelInfo config_direct_convolution_nhwc(const ITensorInfo *src, const ITensorInfo *weights, const PadStrideInfo &conv_info)
{
    // Get GPU target
    GPUTarget gpu_target = CLScheduler::get().target();

    std::unique_ptr<IClDirectConvKernelConfig> t = ClDirectConvKernelConfigurationFactory::create(gpu_target);

    return t->configure(src, weights, conv_info);
}

} // namespace

void ClIndirectConv2d::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                                 const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_LOG_PARAMS(src, weights, biases, dst, conv_info, act_info);

    // Reuse the direct convolution descriptor
    const DirectConvComputeKernelInfo desc = config_direct_convolution_nhwc(src, weights, conv_info);

    // Configure indirect convolution kernels
    auto k0 = std::make_unique<kernels::ClIndirectConv2dAddressPrecalculationKernel>();
    auto k1 = std::make_unique<kernels::ClIndirectConv2dKernel>();

    k0->set_target(CLScheduler::get().target());
    k1->set_target(CLScheduler::get().target());

    k0->configure(compile_context, src, weights, &_indirect_buffer, conv_info, desc);
    k1->configure(compile_context, src, weights, biases, &_indirect_buffer, dst, conv_info, act_info, desc);

    _addr_precalculation_kernel = std::move(k0);
    _indirect_conv_kernel       = std::move(k1);
    _is_prepared                = false;

    // Tune kernels
    CLScheduler::get().tune_kernel_static(*_indirect_conv_kernel);

    // Request memory for the indirect buffer
    _aux_mem[IndirectBuffer] = MemoryInfo(offset_int_vec(IndirectBuffer), MemoryLifetime::Persistent, _indirect_buffer.total_size());
}

Status ClIndirectConv2d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                  const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    // Initialize the direct convolution descriptor
    const DirectConvComputeKernelInfo desc = config_direct_convolution_nhwc(src, weights, conv_info);

    TensorShape ind_buffer_shape = misc::shape_calculator::compute_indirect_buffer_shape(src->tensor_shape(),
                                                                                         src->data_layout(),
                                                                                         weights->tensor_shape(),
                                                                                         conv_info,
                                                                                         desc);

    TensorInfo indirect_buffer(ind_buffer_shape, 1, DataType::S32);

    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClIndirectConv2dAddressPrecalculationKernel::validate(src, weights, &indirect_buffer, conv_info, desc));
    ARM_COMPUTE_RETURN_ON_ERROR(kernels::ClIndirectConv2dKernel::validate(src, weights, biases, &indirect_buffer, dst, conv_info, act_info, desc));

    return Status{};
}

void ClIndirectConv2d::run(ITensorPack &tensors)
{
    CLAuxTensorHandler indirect_buffer(offset_int_vec(IndirectBuffer), _indirect_buffer, tensors, true);

    prepare(tensors);

    ITensorPack indirect_conv2d_pack(tensors);
    indirect_conv2d_pack.add_const_tensor(ACL_SRC_3, indirect_buffer.get());

    // Run indirect convolution
    CLScheduler::get().enqueue_op(*_indirect_conv_kernel, indirect_conv2d_pack, true);
}

void ClIndirectConv2d::prepare(ITensorPack &constants)
{
    if(!_is_prepared)
    {
        ICLTensor *indirect_buffer_aux = utils::cast::polymorphic_downcast<ICLTensor *>(constants.get_tensor(offset_int_vec(IndirectBuffer)));
        ARM_COMPUTE_ERROR_ON(indirect_buffer_aux == nullptr);

        ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Preparing indirect buffer");

        CLAuxTensorHandler indirect_buffer(_indirect_buffer, *indirect_buffer_aux);
        ARM_COMPUTE_ERROR_ON(indirect_buffer.get()->cl_buffer().get() == nullptr);

        ITensorPack indirect_buffer_pack{ { ACL_DST, indirect_buffer.get() } };
        CLScheduler::get().enqueue_op(*_addr_precalculation_kernel, indirect_buffer_pack, true);

        _is_prepared = true;
    }
}

experimental::MemoryRequirements ClIndirectConv2d::workspace() const
{
    return _aux_mem;
}
} // namespace opencl
} // namespace arm_compute
