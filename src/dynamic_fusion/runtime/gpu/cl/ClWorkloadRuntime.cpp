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
#include "arm_compute/dynamic_fusion/runtime/gpu/cl/ClWorkloadRuntime.h"

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "src/dynamic_fusion/runtime/gpu/cl/ClKernelRuntime.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSketchImpl.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSourceCode.h"
#include "support/Cast.h"

#include <algorithm>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
/** Holder of any auxiliary @ref CLTensor required by a @ref GpuWorkloadSourceCode.
 *
 * @note The tensors are not allocated by default, and require the user to explicitly allocate them using the associated @ref TensorInfo and @ref AuxMemoryInfo
 *
 * @note This data holder must remain valid until the @ref ClWorkloadRuntime that uses it, is out of scope
 */
class ClAuxTensors
{
public:
    /** A view of a single auxiliary data and the associated @ref TensorInfo and @ref AuxMemoryInfo
     */
    struct DataView
    {
        DataView() = default;
        DataView(CLTensor *tensor, const TensorInfo &tensor_info, const AuxMemoryInfo &memory_info)
            : tensor{ tensor }, tensor_info{ tensor_info }, memory_info{ memory_info }
        {
        }
        ~DataView()                     = default;
        DataView(const DataView &other) = default;
        DataView &operator=(const DataView &other) = default;
        DataView(DataView &&other)                 = default;
        DataView &operator=(DataView &&other) = default;
        CLTensor     *tensor{};      /**< Pointer to the auxiliary tensor */
        TensorInfo    tensor_info{}; /**< Associated tensor info */
        AuxMemoryInfo memory_info{}; /**< Memory requirement */
    };

    /** Get views of all auxiliary tensors. This is mainly used for allocating the auxiliary tensors. */
    std::vector<DataView> get_tensors()
    {
        return _tensors;
    }
    std::vector<DataView> get_tensors() const
    {
        return _tensors;
    }

    friend Status create_aux_tensors(ClAuxTensors *aux_tensors, const GpuWorkloadSourceCode &code);

private:
    /** Add auxiliary tensor.
     *
     * @param[in] tensor_info @ref ITensorInfo of the auxiliary tensor
     * @param[in] memory_info Memory requirements of the auxiliary tensor
     *
     * @return CLTensor*  Corresponding tensor memory if successfully added, otherwise nullptr
     */
    CLTensor *add_aux_tensor(const ITensorInfo &tensor_info, const AuxMemoryInfo &aux_memory_info)
    {
        const auto t_id             = tensor_info.id();
        auto       find_tensor_pair = _owned_tensors.find(t_id);
        if(find_tensor_pair == _owned_tensors.end())
        {
            return find_tensor_pair->second.get();
        }
        else
        {
            auto tensor        = std::make_unique<CLTensor>();
            auto inserted_pair = _owned_tensors.emplace(t_id, std::move(tensor)).first;
            auto new_tensor    = inserted_pair->second.get();
            _tensors.emplace_back(new_tensor, tensor_info, aux_memory_info);
            return new_tensor;
        }
    }

    std::map<ITensorInfo::Id, std::unique_ptr<CLTensor>> _owned_tensors{};
    std::vector<DataView> _tensors{};
};
/** Construct auxiliary tensors required by @ref GpuWorkloadSourceCode
 *
 * @note This is the only recommended method for user to create @ref ClAuxTensors
 *
 * @param[out] aux_tensors Auxiliary tensors required by the workload code
 * @param[in]  code        @ref GpuWorkloadSourceCode which all tensors bind to
 *
 * @return Status
 */
Status create_aux_tensors(ClAuxTensors *aux_tensors, const GpuWorkloadSourceCode &code)
{
    for(auto t_id : code.tensors())
    {
        // Get tensor object
        const auto workload_arg  = code.query_tensor(t_id);
        ICLTensor *tensor_object = nullptr;
        if(workload_arg->memory_descriptor()->memory_type == MemoryType::Auxiliary)
        {
            // Create aux tensor CLTensor object
            const TensorInfo tensor_info = *workload_arg->tensor_info();
            ARM_COMPUTE_ERROR_ON(tensor_info.id() != t_id);
            const auto aux_memory_info = workload_arg->memory_descriptor()->aux_memory_info;
            tensor_object              = aux_tensors->add_aux_tensor(tensor_info, aux_memory_info);
        }
        if(tensor_object == nullptr)
        {
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Failed to construct an auxiliary tensor");
        }
    }
    return Status{};
}

/** A fast tensor lookup table for runtime tensor objects retrieval
 */
class ClTensorLUT
{
public:
    /** Find a tensor pack associated with the @ref UnitWorkloadId @p uwk_id
     *
     * @param[in] uwk_id @ref UnitWorkloadId associated with the tensor pack
     *
     * @return ITensorPack*
     */
    ITensorPack *find_tensor_pack(UnitWorkloadId uwk_id)
    {
        auto tensor_pack = _tensor_packs.find(uwk_id);
        if(tensor_pack != _tensor_packs.end())
        {
            return &(tensor_pack->second);
        }
        return nullptr;
    }
    /** Get a tensor pack associated with @p uwk_id. Throws a exception if it cannot be found.
     *
     * @param[in] uwk_id @ref UnitWorkloadId associated with the tensor pack
     *
     * @return ITensorPack*
     */
    ITensorPack &get_tensor_pack(UnitWorkloadId uwk_id)
    {
        return _tensor_packs.at(uwk_id);
    }

    friend Status create_tensor_lut(ClTensorLUT *tensor_lut, const GpuWorkloadSourceCode &code, const std::vector<CLTensor *> &user_tensors, const ClAuxTensors &aux_tensors);

private:
    /** Add a tensor pack and associate it with @ref UnitWorkloadId @p uwk_id
     *
     * @param[in] uwk_id      @ref UnitWorkloadId associated with the tensor pack
     * @param[in] tensor_pack Tensor pack to be added
     */
    void add_tensor_pack(UnitWorkloadId uwk_id, const ITensorPack &tensor_pack)
    {
        _tensor_packs[uwk_id] = tensor_pack;
    }
    std::map<UnitWorkloadId, ITensorPack> _tensor_packs{};
};

/** Create a fast tensor lookup table for runtime tensor retrieval
 *
 * @param[out] tensor_lut   @ref ClTensorLUT used by the runtime to feed tensor memories to underlying kernels
 * @param[in]  code         @ref GpuWorkloadSourceCode which all tensors bind to
 * @param[in]  user_tensors User tensors
 * @param[in]  aux_tensors  Auxiliary tensors required by the workload code
 *
 * @return Status
 */
Status create_tensor_lut(ClTensorLUT *tensor_lut, const GpuWorkloadSourceCode &code, const std::vector<CLTensor *> &user_tensors, const ClAuxTensors &aux_tensors)
{
    // Combine user tensors and aux tensors
    std::map<ITensorInfo::Id, CLTensor *> tensor_map;
    for(auto tensor : user_tensors)
    {
        const auto t_id = tensor->info()->id();
        if(tensor_map.find(t_id) != tensor_map.end())
        {
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Clashing tensor ids");
        }
        tensor_map[t_id] = tensor;
    }
    for(const auto &data : aux_tensors.get_tensors())
    {
        const auto t_id   = data.tensor_info.id();
        const auto tensor = data.tensor;
        if(tensor_map.find(t_id) != tensor_map.end())
        {
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Clashing tensor ids");
        }
        tensor_map[t_id] = tensor;
    }

    // Add tensor objects into corresponding tensor packs
    for(auto id_tensor : tensor_map)
    {
        const auto t_id          = id_tensor.first;
        const auto tensor_object = id_tensor.second;
        if(tensor_object == nullptr)
        {
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Trying to add a nullptr into the tensor packs");
        }
        if(tensor_object->allocator()->info().total_size() == 0U)
        {
            return ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "No allocated memory found in tensor");
        }

        for(auto uwk_id : code.get_unit_workloads_from_tensor(t_id))
        {
            ITensorPack *tensor_pack = tensor_lut->find_tensor_pack(uwk_id);
            if(tensor_pack == nullptr)
            {
                tensor_lut->add_tensor_pack(uwk_id, ITensorPack{ { t_id, tensor_object } });
            }
            else
            {
                tensor_pack->add_tensor(t_id, tensor_object);
            }
        }
    }
    return Status{};
}

} // namespace

struct ClWorkloadRuntime::Implementation
{
    std::map<UnitWorkloadId, std::unique_ptr<ClKernelRuntime>> _kernels{};
    std::map<UnitWorkloadId, std::unique_ptr<ClKernelRuntime>> _kernels_prep{};
    bool                  _is_configured{ false };
    bool                  _is_prepared{ false };
    ClTensorLUT           _tensor_lut{};
    ClAuxTensors          _aux_tensors{};
    GpuWorkloadSourceCode _source_code{};
};

ClWorkloadRuntime::ClWorkloadRuntime()
    : _impl{ std::make_unique<Implementation>() }
{
}

ClWorkloadRuntime::~ClWorkloadRuntime() = default;

Status ClWorkloadRuntime::configure(const GpuWorkloadSketch &sketch)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(_impl->_is_configured, "ClWorkloadRuntime cannot be re-configured");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(sketch.gpu_context()->gpu_language() != GpuLanguage::OpenCL, "ClWorkloadRuntime cannot be configured with non-OpenCL workload sketch");
    // Generate source code
    _impl->_source_code = sketch.implementation().generate_source_code();
    // Configure unit workload from source code
    for(auto uwk_id : _impl->_source_code.unit_workloads())
    {
        const auto work  = _impl->_source_code.query_unit_workload(uwk_id);
        const auto stage = work.stage().stage;
        auto       k     = std::make_unique<ClKernelRuntime>();
        k->configure(*sketch.gpu_context()->cl_compile_context(), work.code());

        switch(stage)
        {
            case UnitWorkloadStage::Stage::Run:
                _impl->_kernels.emplace(work.id(), std::move(k));
                break;
            case UnitWorkloadStage::Stage::Prepare:
                _impl->_kernels_prep.emplace(work.id(), std::move(k));
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid unit workload stage");
        }
        break;
    }
    // Create auxiliary tensor objects
    create_aux_tensors(&_impl->_aux_tensors, _impl->_source_code);
    _impl->_is_configured = true;
    return Status{};
}

void ClWorkloadRuntime::prepare()
{
    if(!_impl->_is_prepared)
    {
        for(auto &id_kernel_pair : _impl->_kernels_prep)
        {
            const bool flush_queue = false;
            const auto uwk_id      = id_kernel_pair.first;
            auto       kernel      = id_kernel_pair.second.get();
            CLScheduler::get().enqueue_op(*kernel, _impl->_tensor_lut.get_tensor_pack(uwk_id), flush_queue);
        }

        _impl->_is_prepared = true;
    }
}

Status ClWorkloadRuntime::run(const std::vector<CLTensor *> &tensors)
{
    // Need to create the tensor lut in every run, unless the user can guarantee the binding remains fixed,
    // in which case the lut can be cached during prepare
    const auto st = create_tensor_lut(&_impl->_tensor_lut, _impl->_source_code, tensors, _impl->_aux_tensors);
    ARM_COMPUTE_RETURN_ON_ERROR(st);
    prepare();
    for(auto &id_kernel_pair : _impl->_kernels)
    {
        // Flush the command queue on the last kernel
        const bool flush_queue = false;
        const auto uwk_id      = id_kernel_pair.first;
        auto       kernel      = id_kernel_pair.second.get();
        CLScheduler::get().enqueue_op(*kernel, _impl->_tensor_lut.get_tensor_pack(uwk_id), flush_queue);
    }
    return Status{};
}

std::vector<std::pair<CLTensor *, AuxMemoryInfo>> ClWorkloadRuntime::get_auxiliary_tensors()
{
    std::vector<std::pair<CLTensor *, AuxMemoryInfo>> aux_tensors;
    for(const auto &data : _impl->_aux_tensors.get_tensors())
    {
        aux_tensors.emplace_back(data.tensor, data.memory_info);
    }
    return aux_tensors;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
