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
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/gpu/cl/operators/ClFullyConnected.h"

namespace arm_compute
{
using namespace arm_compute::experimental;

struct CLFullyConnectedLayer::Impl
{
    MemoryGroup      memory_group{};
    IWeightsManager *weights_manager{ nullptr };

    std::unique_ptr<opencl::ClFullyConnected> op{ nullptr };

    const ITensor *original_weights{ nullptr };

    ITensorPack                      run_pack{};
    WorkspaceData<CLTensor>          workspace{};
    experimental::MemoryRequirements aux_mem_req{};

    bool is_prepared{ false };
};

CLFullyConnectedLayer::CLFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager, IWeightsManager *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group    = MemoryGroup(std::move(memory_manager));
    _impl->weights_manager = weights_manager;
}

CLFullyConnectedLayer::~CLFullyConnectedLayer() = default;

void CLFullyConnectedLayer::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                      FullyConnectedLayerInfo fc_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, fc_info);
}

void CLFullyConnectedLayer::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                      FullyConnectedLayerInfo fc_info)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLFullyConnectedLayer::validate(input->info(),
                                                               weights->info(),
                                                               biases != nullptr ? biases->info() : nullptr,
                                                               output->info(),
                                                               fc_info));

    _impl->op               = std::make_unique<opencl::ClFullyConnected>();
    _impl->original_weights = weights;
    _impl->is_prepared      = fc_info.retain_internal_weights;

    _impl->op->configure(compile_context, input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), fc_info);

    if(_impl->weights_manager != nullptr)
    {
        _impl->weights_manager->manage(weights);
    }

    if(!_impl->is_prepared)
    {
        _impl->aux_mem_req = _impl->op->workspace();
        _impl->run_pack    = { { ACL_SRC_0, input }, { ACL_SRC_1, weights }, { ACL_SRC_2, biases }, { ACL_DST, output } };
        _impl->workspace   = manage_workspace<CLTensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->run_pack);
    }
    else
    {
        _impl->run_pack.add_tensor(ACL_SRC_0, input);
        _impl->run_pack.add_tensor(ACL_DST, output);
    }
}

Status CLFullyConnectedLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                       FullyConnectedLayerInfo fc_info)
{
    return opencl::ClFullyConnected::validate(input, weights, biases, output, fc_info);
}

void CLFullyConnectedLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);
    _impl->op->run(_impl->run_pack);
}

void CLFullyConnectedLayer::prepare()
{
    if(!_impl->is_prepared)
    {
        _impl->op->prepare(_impl->run_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<CLTensor>(_impl->aux_mem_req, _impl->workspace);
        _impl->is_prepared = true;

        // Handle weights managed infrastructure
        if(_impl->weights_manager != nullptr && _impl->weights_manager->are_weights_managed(_impl->original_weights))
        {
            // If function marks b as unused ensure that all prepare stages are done before releasing
            const ITensor *original_b = _impl->original_weights;
            if(!original_b->is_used())
            {
                _impl->weights_manager->mark_as_unused(original_b);
            }
            _impl->original_weights->mark_as_used();
            _impl->weights_manager->release(_impl->original_weights);
        }
    }
}
} // namespace arm_compute
