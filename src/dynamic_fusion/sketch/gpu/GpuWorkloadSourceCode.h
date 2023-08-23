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
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadContextImpl.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
#ifdef ACL_INTERNAL_TEST_CKW_IN_DF
namespace
{
/** Extract kernel arguments of one tensor from a flat list of kernel arguments.
 *
 * @param[in] flat_kernel_args
 * @return GpuKernelArgumentList
 */
GpuKernelArgumentList extract_kernel_args_for_one_tensor(GpuKernelArgumentList &flat_kernel_args)
{
    if(flat_kernel_args.empty())
    {
        return {};
    }
    GpuKernelArgumentList tensor_kargs{};

    const GpuKernelArgumentBinding &karg_head = flat_kernel_args.front();
    tensor_kargs.push_back(karg_head);
    flat_kernel_args.pop_front();
    const auto tensor_id = karg_head.id();

    while(!flat_kernel_args.empty())
    {
        const GpuKernelArgumentBinding &karg = flat_kernel_args.front();
        if(karg.id() != tensor_id) // Encounter the next tensor, return the current tensor's kernel arguments
        {
            return tensor_kargs;
        }
        tensor_kargs.push_back(karg);
        flat_kernel_args.pop_front();
    }
    return tensor_kargs;
}
}
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
/** Uniquely identifies a @ref GpuUnitWorkload within a @ref GpuWorkloadSourceCode */
using UnitWorkloadId = int32_t;

/** Describes all the info related to a **workload argument** (tensor) in order to:
 *  - be used by runtime to configure gpu kernel argument
 *  - be used by memory managers to allocate required memory
 */
class GpuWorkloadArgument
{
public:
    /** Default constructor */
    GpuWorkloadArgument() = default;
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
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
#else  // ACL_INTERNAL_TEST_CKW_IN_DF
    /** Constructor
     *
     * @param[in] tensor_info     @ref ITensorInfo of the workload argument
     * @param[in] mem_desc        @ref MemoryDescriptor of the workload argument
     * @param[in] kernel_arg_list @ref GpuKernelArgumentList of the workload argument
     */
    GpuWorkloadArgument(const ITensorInfo           &tensor_info,
                        const MemoryDescriptor      &mem_desc,
                        const GpuKernelArgumentList &kernel_args)
        : _tensor_info{ tensor_info },
          _mem_desc{ mem_desc },
          _kernel_args{ kernel_args }
    {
    }
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
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
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
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
#else  // ACL_INTERNAL_TEST_CKW_IN_DF
    /** Get @ref GpuKernelArgumentList of the workload tensor */
    GpuKernelArgumentList *kernel_argument_list()
    {
        return &_kernel_args;
    }
    /** Get @ref GpuKernelArgumentList of the workload tensor */
    const GpuKernelArgumentList *kernel_argument_list() const
    {
        return &_kernel_args;
    }
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
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
    TensorInfo       _tensor_info{};
    MemoryDescriptor _mem_desc{};
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
    GpuKernelArgumentInfo _kernel_arg_info {};
#else  // ACL_INTERNAL_TEST_CKW_IN_DF
    GpuKernelArgumentList     _kernel_args {};
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
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
     * @param[in] context     @ref GpuWorkloadContext associated with the unit workload
     *
     * @return UnitWorkloadId  Allocated unit workload id
     */
    UnitWorkloadId add_unit_workload(const GpuKernelSourceCode &kernel_code, const UnitWorkloadStage &stage, const MemoryDescriptorMap &mem_map, const GpuWorkloadContext *context)
    {
        // Use the size of the kernel codes as Id
        const auto uwk_id    = static_cast<UnitWorkloadId>(_unit_workloads.size());
        const auto unit_work = GpuUnitWorkload(uwk_id, kernel_code, stage);
        _unit_workloads.push_back(unit_work);
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
        ARM_COMPUTE_UNUSED(context);
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
#else  // ACL_INTERNAL_TEST_CKW_IN_DF
        GpuKernelArgumentList flat_kernel_args = kernel_code.arguments();
        GpuKernelArgumentList tensor_kargs{};
        while(true)
        {
            tensor_kargs = extract_kernel_args_for_one_tensor(flat_kernel_args);
            if(tensor_kargs.empty())
            {
                break;
            }
            else
            {
                const auto tensor_id           = tensor_kargs.at(0).id();
                _workload_arguments[tensor_id] = GpuWorkloadArgument{ *context->implementation().get_tensor_info(tensor_id), mem_map.at(tensor_id), tensor_kargs };
                if(_tensor_uwork_map.find(tensor_id) == _tensor_uwork_map.end())
                {
                    _tensor_uwork_map[tensor_id] = std::set<UnitWorkloadId>();
                }
                _tensor_uwork_map[tensor_id].insert(uwk_id);
            }
        }
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
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
