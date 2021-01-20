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
#include "src/runtime/NEON/functions/NEPoolingAssemblyDispatch.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/assembly/NEPoolingAssemblyWrapperKernel.h"

namespace arm_compute
{
namespace experimental
{
NEPoolingAssemblyDispatch::~NEPoolingAssemblyDispatch() = default;

void NEPoolingAssemblyDispatch::configure(const ITensorInfo *input, ITensorInfo *output, const PoolingLayerInfo &info)
{
    const CPUInfo     &ci          = NEScheduler::get().cpu_info();
    const unsigned int num_threads = NEScheduler::get().num_threads();

    // If we don't support a combination of data types, silently return: it is the caller's responsibility to check if configure() was successful via is_configured()
    if(!NEPoolingAssemblyDispatch::validate(input, output, info))
    {
        return;
    }

    auto pooling_wrapper = std::make_unique<NEPoolingAssemblyWrapperKernel>();
    ARM_COMPUTE_ERROR_ON(pooling_wrapper == nullptr);
    pooling_wrapper->configure(input, output, info, ci);

    // Check if we have Global Pooling Layer
    _is_global_pooling_layer = (input->dimension(2) == info.pool_size.width) && (input->dimension(1) == info.pool_size.height);

    // Set workspace requirements
    const unsigned int alignment = 4096;
    _workspace.push_back(MemoryInfo(TensorType::ACL_DST_1, pooling_wrapper->get_working_size(num_threads), alignment));

    _kernel = std::move(pooling_wrapper);
}

Status NEPoolingAssemblyDispatch::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &info)
{
    return NEPoolingAssemblyWrapperKernel::validate(input, output, info);
}

bool NEPoolingAssemblyDispatch::is_configured() const
{
    return _kernel != nullptr;
}

void NEPoolingAssemblyDispatch::run(ITensorPack &tensors)
{
    if(tensors.empty())
    {
        ARM_COMPUTE_ERROR("No inputs provided");
    }

    if(_is_global_pooling_layer)
    {
        NEScheduler::get().schedule_op(_kernel.get(), Window::DimX, _kernel->window(), tensors);
    }
    else
    {
        NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
    }
}
} // namespace experimental

struct NEPoolingAssemblyDispatch::Impl
{
    const ITensor                                           *src{ nullptr };
    ITensor                                                 *dst{ nullptr };
    ITensor                                                 *workspace{ nullptr };
    std::unique_ptr<experimental::NEPoolingAssemblyDispatch> op{ nullptr };
};

NEPoolingAssemblyDispatch::NEPoolingAssemblyDispatch(NEPoolingAssemblyDispatch &&) = default;

NEPoolingAssemblyDispatch &NEPoolingAssemblyDispatch::operator=(NEPoolingAssemblyDispatch &&) = default;

NEPoolingAssemblyDispatch::~NEPoolingAssemblyDispatch() = default;

NEPoolingAssemblyDispatch::NEPoolingAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>()),
      _memory_group(std::move(memory_manager)),
      _workspace()
{
}

void NEPoolingAssemblyDispatch::configure(const ITensor *input, ITensor *output, const PoolingLayerInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _impl->src       = input;
    _impl->dst       = output;
    _impl->workspace = &_workspace;

    _impl->op = std::make_unique<experimental::NEPoolingAssemblyDispatch>();
    _impl->op->configure(input->info(), output->info(), info);

    const auto workspace = _impl->op->workspace().at(0);
    if(workspace.size > 0)
    {
        // Allocate workspace
        allocate_workspace(workspace.size, workspace.alignment);
    }
}

Status NEPoolingAssemblyDispatch::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &info)
{
    return experimental::NEPoolingAssemblyDispatch::validate(input, output, info);
}

bool NEPoolingAssemblyDispatch::is_configured() const
{
    return _impl->op->is_configured();
}

void NEPoolingAssemblyDispatch::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST_0, _impl->dst);
    pack.add_tensor(TensorType::ACL_DST_1, _impl->workspace);
    _impl->op->run(pack);
}

void NEPoolingAssemblyDispatch::allocate_workspace(size_t workspace_size, size_t alignment)
{
    ARM_COMPUTE_ERROR_ON_MSG(workspace_size == 0, "size cannot be 0");
    _workspace.allocator()->init(TensorInfo(TensorShape{ (workspace_size + alignment /* FIXME: remove alignment after COMPMID-1088 */) }, 1, DataType::S8), alignment);
    _memory_group.manage(&_workspace);
    _workspace.allocator()->allocate();
}
} //namespace arm_compute
