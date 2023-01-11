/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSOURCECODE
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSOURCECODE

#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/dynamic_fusion/sketch/MemoryDescriptor.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelSourceCode.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Uniquely identifies a @ref GpuUnitWorkload within a @ref GpuWorkloadSourceCode */
using UnitWorkloadId = int32_t;

/** Describes all the info related to a kernel in order to:
 *  - be used by runtime to configure gpu kernel argument
 *  - be used by memory managers to allocate required memory
 */
class GpuWorkloadArgument
{
public:
    /** Default constructor */
    GpuWorkloadArgument() = default;
    /** Constructor
     *
     * @param[in] tensor_info     @ref ITensorInfo of the workload argument
     * @param[in] mem_desc        @ref MemoryDescriptor of the workload argument
     * @param[in] kernel_arg_info @ref GpuKernelArgumentInfo of the workload argument
     */
    GpuWorkloadArgument(const ITensorInfo           &tensor_info,
                        const MemoryDescriptor      &mem_desc,
                        const GpuKernelArgumentInfo &kernel_arg_info)
        : _tensor_info{ tensor_info },
          _mem_desc{ mem_desc },
          _kernel_arg_info{ kernel_arg_info }
    {
    }
    /** Get tensor id within workload */
    ITensorInfo::Id id() const
    {
        return _tensor_info.id();
    }
    /** Get @ref ITensorInfo of the argument */
    ITensorInfo *tensor_info()
    {
        return &_tensor_info;
    }
    /** Get @ref ITensorInfo of the argument */
    const ITensorInfo *tensor_info() const
    {
        return &_tensor_info;
    }
    /** Get @ref MemoryDescriptor of the argument */
    MemoryDescriptor *memory_descriptor()
    {
        return &_mem_desc;
    }
    /** Get @ref MemoryDescriptor of the argument */
    const MemoryDescriptor *memory_descriptor() const
    {
        return &_mem_desc;
    }
    /** Get @ref GpuKernelArgumentInfo of the argument */
    GpuKernelArgumentInfo *kernel_argument_info()
    {
        return &_kernel_arg_info;
    }
    /** Get @ref GpuKernelArgumentInfo of the argument */
    const GpuKernelArgumentInfo *kernel_argument_info() const
    {
        return &_kernel_arg_info;
    }
    /** Check if the workload argument has valid id
     *
     * @return true   If has valid id
     * @return false  Otherwise
     */
    bool has_valid_id() const
    {
        return _tensor_info.has_valid_id();
    }

private:
    TensorInfo            _tensor_info{};
    MemoryDescriptor      _mem_desc{};
    GpuKernelArgumentInfo _kernel_arg_info{};
};

/** Describes when a unit workload is run.
 */
struct UnitWorkloadStage
{
    enum class Stage
    {
        Prepare, /**< Only run once at the beginning. */
        Run,     /**< Run every time after the first time. */
    };
    Stage stage{ Stage::Run };
};

inline bool operator==(const UnitWorkloadStage &stage0, const UnitWorkloadStage &stage1)
{
    return stage0.stage == stage1.stage;
}

/** The atomic unit in a Gpu workload. It contains exactly one kernel to run.
 */
class GpuUnitWorkload
{
public:
    /** Default constructor */
    GpuUnitWorkload() = default;
    /** Constructor
     *
     * @param[in] id          Id that uniquely identifies this unit workload in a workload
     * @param[in] kernel_code @ref GpuKernelSourceCode contained within
     * @param[in] stage       Stage of the unit workload
     */
    GpuUnitWorkload(UnitWorkloadId id, const GpuKernelSourceCode &kernel_code, const UnitWorkloadStage &stage)
        : _id{ id }, _kernel_code{ kernel_code }, _stage{ stage }
    {
    }
    /** Get the id of the unit workload */
    UnitWorkloadId id() const
    {
        return _id;
    }
    /** Get reference to the underlying @ref GpuKernelSourceCode */
    const GpuKernelSourceCode &code() const
    {
        return _kernel_code;
    }
    /** Get the stage of the unit workload */
    UnitWorkloadStage stage() const
    {
        return _stage;
    }

private:
    UnitWorkloadId      _id{};
    GpuKernelSourceCode _kernel_code{};
    UnitWorkloadStage   _stage{};
};

/** Hold the generated kernel source code and other information required to compile and run the workload.
 */
class GpuWorkloadSourceCode
{
public:
    /** Default constructor */
    GpuWorkloadSourceCode() = default;
    /** Add a unit workload to the workload code
     *
     * @param[in] kernel_code @ref GpuKernelSourceCode to be contained within the unit workload
     * @param[in] stage       Stage of the unit workload
     * @param[in] mem_map     @ref MemoryDescriptor map for all tensors within the unit workload
     *
     * @return UnitWorkloadId  Allocated unit workload id
     */
    UnitWorkloadId add_unit_workload(const GpuKernelSourceCode &kernel_code, const UnitWorkloadStage &stage, const MemoryDescriptorMap &mem_map)
    {
        // Use the size of the kernel codes as Id
        const auto uwk_id    = static_cast<UnitWorkloadId>(_unit_workloads.size());
        const auto unit_work = GpuUnitWorkload(uwk_id, kernel_code, stage);
        _unit_workloads.push_back(unit_work);
        // Assemble kernel argument with memory descriptor to form workload argument
        for(const auto &id_arg : kernel_code.arguments())
        {
            const auto arg_id           = id_arg.first;
            const auto arg              = id_arg.second;
            _workload_arguments[arg_id] = GpuWorkloadArgument{ *arg.tensor_info(), mem_map.at(arg_id), *arg.kernel_argument_info() };
            if(_tensor_uwork_map.find(arg_id) == _tensor_uwork_map.end())
            {
                _tensor_uwork_map[arg_id] = std::set<UnitWorkloadId>();
            }
            _tensor_uwork_map[arg_id].insert(uwk_id);
        }
        return uwk_id;
    }
    /** Get a unit workload from its id */
    const GpuUnitWorkload &query_unit_workload(UnitWorkloadId id) const
    {
        ARM_COMPUTE_ERROR_ON(id < 0);
        return _unit_workloads.at(id);
    }
    /** Get all unit workloads sorted in topological order */
    std::vector<UnitWorkloadId> unit_workloads() const
    {
        std::vector<UnitWorkloadId> ids{};

        for(const auto &uwk : _unit_workloads)
        {
            ids.push_back(uwk.id());
        }
        return ids;
    }
    /** Get a @ref GpuWorkloadArgument from its associated tensor id */
    const GpuWorkloadArgument *query_tensor(ITensorInfo::Id t_id) const
    {
        return &_workload_arguments.at(t_id);
    }
    /** Get all tensors in the entire workload */
    std::vector<ITensorInfo::Id> tensors() const
    {
        std::vector<ITensorInfo::Id> ids{};
        for(const auto &id_tensor : _workload_arguments)
        {
            ids.push_back(id_tensor.first);
        }
        return ids;
    }
    /** Get all unit workloads connected to the tensor with @p t_id */
    std::vector<UnitWorkloadId> get_unit_workloads_from_tensor(ITensorInfo::Id t_id) const
    {
        const auto unit_work_set = _tensor_uwork_map.at(t_id);
        return std::vector<UnitWorkloadId>(unit_work_set.begin(), unit_work_set.end());
    }

private:
    std::vector<GpuUnitWorkload> _unit_workloads{};
    std::map<ITensorInfo::Id, GpuWorkloadArgument>      _workload_arguments{};
    std::map<ITensorInfo::Id, std::set<UnitWorkloadId>> _tensor_uwork_map{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADSOURCECODE */
