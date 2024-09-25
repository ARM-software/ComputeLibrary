/*
 * Copyright (c) 2021, 2024 Arm Limited.
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
#ifndef ACL_SRC_CORE_HELPERS_MEMORYHELPERS_H
#define ACL_SRC_CORE_HELPERS_MEMORYHELPERS_H

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>
#include <utility>
#include <vector>

namespace arm_compute
{
inline int offset_int_vec(int offset)
{
    return ACL_INT_VEC + offset;
}

template <typename TensorType>
struct WorkspaceDataElement
{
    int                          slot{-1};
    experimental::MemoryLifetime lifetime{experimental::MemoryLifetime::Temporary};
    std::unique_ptr<TensorType>  tensor{nullptr};
};

template <typename TensorType>
using WorkspaceData = std::vector<WorkspaceDataElement<TensorType>>;

template <typename TensorType>
WorkspaceData<TensorType>
manage_workspace(const experimental::MemoryRequirements &mem_reqs, MemoryGroup &mgroup, ITensorPack &run_pack)
{
    ITensorPack dummy_pack = ITensorPack();
    return manage_workspace<TensorType>(mem_reqs, mgroup, run_pack, dummy_pack);
}

template <typename TensorType>
WorkspaceData<TensorType> manage_workspace(const experimental::MemoryRequirements &mem_reqs,
                                           MemoryGroup                            &mgroup,
                                           ITensorPack                            &run_pack,
                                           ITensorPack                            &prep_pack,
                                           bool                                    allocate_now = true)
{
    WorkspaceData<TensorType> workspace_memory;
    for (const auto &req : mem_reqs)
    {
        if (req.size == 0)
        {
            continue;
        }

        const auto aux_info = TensorInfo{TensorShape(req.size), 1, DataType::U8};
        workspace_memory.emplace_back(
            WorkspaceDataElement<TensorType>{req.slot, req.lifetime, std::make_unique<TensorType>()});

        auto aux_tensor = workspace_memory.back().tensor.get();
        ARM_COMPUTE_ERROR_ON_NULLPTR(aux_tensor);
        aux_tensor->allocator()->init(aux_info, req.alignment);

        if (req.lifetime == experimental::MemoryLifetime::Temporary)
        {
            mgroup.manage(aux_tensor);
        }
        else
        {
            prep_pack.add_tensor(req.slot, aux_tensor);
        }
        run_pack.add_tensor(req.slot, aux_tensor);
    }

    for (auto &mem : workspace_memory)
    {
        if (allocate_now || mem.lifetime == experimental::MemoryLifetime::Temporary)
        {
            auto tensor = mem.tensor.get();
            tensor->allocator()->allocate();
        }
    }

    return workspace_memory;
}

template <typename TensorType>
void release_prepare_tensors(WorkspaceData<TensorType> &workspace, ITensorPack &prep_pack)
{
    workspace.erase(std::remove_if(workspace.begin(), workspace.end(),
                                   [&prep_pack](auto &wk)
                                   {
                                       const bool to_erase = wk.lifetime == experimental::MemoryLifetime::Prepare;
                                       if (to_erase)
                                       {
                                           prep_pack.remove_tensor(wk.slot);
                                       }
                                       return to_erase;
                                   }),
                    workspace.end());
}

/** Allocate all tensors with Persistent or Prepare lifetime if not already allocated */
template <typename TensorType>
void allocate_tensors(const experimental::MemoryRequirements &mem_reqs, WorkspaceData<TensorType> &workspace)
{
    for (auto &ws : workspace)
    {
        const int slot = ws.slot;
        for (auto &m : mem_reqs)
        {
            if (m.slot == slot && m.lifetime != experimental::MemoryLifetime::Temporary)
            {
                auto tensor = ws.tensor.get();
                if (!tensor->allocator()->is_allocated())
                {
                    tensor->allocator()->allocate();
                }
                break;
            }
        }
    }
}

/** Utility function to release tensors with lifetime marked as Prepare */
template <typename TensorType>
void release_temporaries(const experimental::MemoryRequirements &mem_reqs, WorkspaceData<TensorType> &workspace)
{
    for (auto &ws : workspace)
    {
        const int slot = ws.slot;
        for (auto &m : mem_reqs)
        {
            if (m.slot == slot && m.lifetime == experimental::MemoryLifetime::Prepare)
            {
                auto tensor = ws.tensor.get();
                tensor->allocator()->free();
                break;
            }
        }
    }
}
} // namespace arm_compute
#endif // ACL_SRC_CORE_HELPERS_MEMORYHELPERS_H
