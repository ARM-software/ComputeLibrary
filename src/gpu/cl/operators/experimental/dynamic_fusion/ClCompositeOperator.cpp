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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#include "arm_compute/runtime/experimental/ClCompositeOperator.h"

#include "arm_compute/core/experimental/ClWorkload.h"
#include "arm_compute/core/experimental/Types.h"
#include "src/gpu/cl/kernels/experimental/dynamic_fusion/ClCompositeKernel.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
Status add_tensor_to_tensor_pack(int wk_tensor_id, ICLTensor *tensor, const ClWorkload &workload, TensorPackMap &prepare_pack_map, TensorPackMap &run_pack_map)
{
    if(tensor == nullptr)
    {
        return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Trying to add a nullptr into the tensor packs");
    }
    const auto                          bp_tensor_id = workload.tensors.at(wk_tensor_id).kernel_arg.arg_id; // blueprint tensor id
    std::vector<ClWorkload::UnitWorkId> uwk_ids{};
    const auto                          src_uwk_ids = workload.graph.src_ops_from_tensor(wk_tensor_id);
    const auto                          dst_uwk_ids = workload.graph.dst_ops_from_tensor(wk_tensor_id);
    uwk_ids.insert(uwk_ids.end(), src_uwk_ids.begin(), src_uwk_ids.end());
    uwk_ids.insert(uwk_ids.end(), dst_uwk_ids.begin(), dst_uwk_ids.end());

    for(auto uwk_id : uwk_ids)
    {
        TensorPackMap *pack_map  = nullptr;
        const auto     uwk_stage = workload.unit_workloads.at(uwk_id).stage.stage;
        switch(uwk_stage)
        {
            case UnitWorkloadStage::Stage::Run:
                pack_map = &run_pack_map;
                break;
            case UnitWorkloadStage::Stage::Prepare:
                pack_map = &prepare_pack_map;
                break;
            default:
                return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported workload stage");
        }

        ITensorPack *tensor_pack = pack_map->find_tensor_pack(uwk_id);
        if(tensor_pack == nullptr)
        {
            pack_map->add_tensor_pack(uwk_id, ITensorPack{ { bp_tensor_id, tensor } });
        }
        else
        {
            tensor_pack->add_tensor(bp_tensor_id, tensor);
        }
    }
    return Status{};
}

} // namespace

ITensorPack *TensorPackMap::find_tensor_pack(UnitWorkload::Id uwk_id)
{
    auto tensor_pack = _tensor_packs.find(uwk_id);
    if(tensor_pack != _tensor_packs.end())
    {
        return &(tensor_pack->second);
    }
    return nullptr;
}

ITensorPack &TensorPackMap::get_tensor_pack(UnitWorkload::Id uwk_id)
{
    return _tensor_packs.at(uwk_id);
}

void TensorPackMap::add_tensor_pack(UnitWorkload::Id uwk_id, const ITensorPack &tensor_pack)
{
    _tensor_packs[uwk_id] = tensor_pack;
}

Status bind_tensors(ClAuxTensorData &aux_tensor_data, TensorPackMap &prepare_pack_map, TensorPackMap &run_pack_map, const ClWorkload &workload, const OpTensorBinding &op_tensors)
{
    for(auto tensor : workload.tensors)
    {
        const auto wk_tensor_id  = tensor.first; // workload tensor id
        ICLTensor *tensor_object = nullptr;
        if(tensor.second.memory_type == MemoryType::Core)
        {
            const auto op_tensor_id   = workload.op_tensor_id_lut.at(wk_tensor_id);
            auto       op_tensor_find = op_tensors.find(op_tensor_id);
            if(op_tensor_find == op_tensors.end())
            {
                return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Cannot find binding for some operator tensor");
            }
            tensor_object = utils::cast::polymorphic_downcast<ICLTensor *>(op_tensor_find->second);
        }
        else if(tensor.second.memory_type == MemoryType::Auxiliary)
        {
            // Create aux tensor CLTensor object
            const TensorInfo tensor_info = *tensor.second.info;
            const auto       memory_info = tensor.second.memory_info;
            tensor_object                = aux_tensor_data.add_aux_tensor(wk_tensor_id, tensor_info, memory_info);
        }
        else
        {
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Unsupported tensor memory type");
        }

        const auto st = add_tensor_to_tensor_pack(wk_tensor_id, tensor_object, workload, prepare_pack_map, run_pack_map);
        ARM_COMPUTE_RETURN_ON_ERROR(st);
    }
    return Status{};
}

CLTensor *ClAuxTensorData::add_aux_tensor(int tensor_id, const ITensorInfo &tensor_info, const AuxMemoryInfo &memory_info)
{
    auto find_tensor_pair = _owned_tensors.find(tensor_id);
    if(find_tensor_pair == _owned_tensors.end())
    {
        return find_tensor_pair->second.get();
    }
    else
    {
        auto tensor        = std::make_unique<CLTensor>();
        auto inserted_pair = _owned_tensors.emplace(tensor_id, std::move(tensor)).first;
        auto new_tensor    = inserted_pair->second.get();
        _tensors.emplace_back(new_tensor, tensor_info, memory_info);
        return new_tensor;
    }
}

std::vector<ClAuxTensorData::DataView> &ClAuxTensorData::get_tensors()
{
    return _tensors;
}
struct ClCompositeOperator::Implementation
{
    std::map<UnitWorkload::Id, std::unique_ptr<ClCompositeKernel>> _kernels{};
    std::map<UnitWorkload::Id, std::unique_ptr<ClCompositeKernel>> _kernels_prep{};
    ClWorkload _workload{};
    bool       _is_prepared{ false };
};

ClCompositeOperator::ClCompositeOperator()
    : _impl{ std::make_unique<Implementation>() }
{
}

ClCompositeOperator::~ClCompositeOperator() = default;

void ClCompositeOperator::configure(const CLCompileContext &ctx, const ClWorkload &workload)
{
    ARM_COMPUTE_ERROR_THROW_ON(ClCompositeOperator::validate(workload));
    _impl->_workload = workload;

    // Traverse workloads in topological order
    const auto sorted = workload.graph.topological_sort().second;
    for(const auto &node : sorted)
    {
        auto work  = workload.unit_workloads.at(node.op);
        auto stage = work.stage.stage;
        auto k     = std::make_unique<ClCompositeKernel>();
        k->configure(ctx, work.code);

        switch(stage)
        {
            case UnitWorkloadStage::Stage::Run:
                _impl->_kernels.emplace(work.id, std::move(k));
                break;
            case UnitWorkloadStage::Stage::Prepare:
                _impl->_kernels_prep.emplace(work.id, std::move(k));
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid stage");
        }
        break;
    }
}

Status ClCompositeOperator::validate(const ClWorkload &workload)
{
    return workload.status;
}

void ClCompositeOperator::prepare(TensorPackMap &tensor_pack_map)
{
    if(!_impl->_is_prepared)
    {
        for(auto &id_kernel_pair : _impl->_kernels_prep)
        {
            const bool flush_queue = false;
            const auto uwk_id      = id_kernel_pair.first;
            auto       kernel      = id_kernel_pair.second.get();
            CLScheduler::get().enqueue_op(*kernel, tensor_pack_map.get_tensor_pack(uwk_id), ClExecutionDescriptor{}, flush_queue);
        }

        _impl->_is_prepared = true;
    }
}

void ClCompositeOperator::run(TensorPackMap &tensor_pack_map)
{
    ARM_COMPUTE_ERROR_ON_MSG(!_impl->_is_prepared, "Operator is not prepared");

    for(auto &id_kernel_pair : _impl->_kernels)
    {
        // Flush the command queue on the last kernel
        const bool flush_queue = false;
        const auto uwk_id      = id_kernel_pair.first;
        auto       kernel      = id_kernel_pair.second.get();
        CLScheduler::get().enqueue_op(*kernel, tensor_pack_map.get_tensor_pack(uwk_id), ClExecutionDescriptor{}, flush_queue);
    }
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */