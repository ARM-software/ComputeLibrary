/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/operators/ClGemmConv2d.h"
#include "support/Cast.h"

#include <cmath>
#include <memory>
#include <tuple>

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::utils::cast;
using namespace arm_compute::experimental;

struct CLGEMMConvolutionLayer::Impl
{
    const ITensor                        *weights{ nullptr };
    std::unique_ptr<opencl::ClGemmConv2d> op{ nullptr };
    ITensorPack                           run_pack{};
    ITensorPack                           prep_pack{};
    MemoryGroup                           memory_group{};
    IWeightsManager                      *weights_manager{ nullptr };
    MemoryRequirements                    aux_mem_req{};
    WorkspaceData<CLTensor>               workspace_tensors{};
    bool                                  is_prepared{ false };
};

CLGEMMConvolutionLayer::CLGEMMConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager, IWeightsManager *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group    = MemoryGroup(memory_manager);
    _impl->weights_manager = weights_manager;
}

CLGEMMConvolutionLayer::~CLGEMMConvolutionLayer() = default;

void CLGEMMConvolutionLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                       const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, weights_info, dilation, act_info, num_groups);
}

void CLGEMMConvolutionLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                       const PadStrideInfo &conv_info,
                                       const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    _impl->weights               = weights;
    _impl->op                    = std::make_unique<opencl::ClGemmConv2d>();
    const Conv2dInfo conv2d_info = Conv2dInfo(conv_info, dilation, act_info, false, num_groups);
    _impl->op->configure(compile_context, input->info(), weights->info(), (biases != nullptr ? biases->info() : nullptr), output->info(), conv2d_info, weights_info);

    _impl->run_pack =
    {
        { TensorType::ACL_SRC_0, input },
        { TensorType::ACL_SRC_1, weights },
        { TensorType::ACL_SRC_2, biases },
        { TensorType::ACL_DST, output }
    };
    _impl->prep_pack =
    {
        { TensorType::ACL_SRC_1, weights },
        { TensorType::ACL_SRC_2, biases },
    };
    _impl->aux_mem_req       = _impl->op->workspace();
    _impl->workspace_tensors = manage_workspace<CLTensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
}

Status CLGEMMConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                        const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    const Conv2dInfo conv2d_info = Conv2dInfo(conv_info, dilation, act_info, false, num_groups);
    return opencl::ClGemmConv2d::validate(input, weights, biases, output, conv2d_info, weights_info);
}

void CLGEMMConvolutionLayer::run()
{
    prepare();
    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    _impl->op->run(_impl->run_pack);
}

void CLGEMMConvolutionLayer::prepare()
{
    if(!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->prep_pack);
        auto has_reshape = std::find_if(_impl->aux_mem_req.begin(),
                                        _impl->aux_mem_req.end(),
                                        [](const MemoryInfo & m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

        if(has_reshape != std::end(_impl->aux_mem_req))
        {
            _impl->weights->mark_as_unused();
        }
        else
        {
            // Pack the B matrix to be used as the underlying GEMM performs no reshapes
            _impl->run_pack.add_const_tensor(ACL_SRC_1, _impl->weights);
        }
        release_temporaries(_impl->aux_mem_req, _impl->workspace_tensors);
        _impl->is_prepared = true;
    }
}
} // namespace arm_compute
