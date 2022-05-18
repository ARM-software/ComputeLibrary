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
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IWORKLOAD_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IWORKLOAD_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/experimental/Types.h"

#include "arm_compute/core/experimental/DependencyGraph.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Describes when a Unit Workload is run.
 *
 */
struct UnitWorkloadStage
{
    enum class Stage
    {
        Prepare, /**< Only run once at the beginning. */
        Run,     /**< Run every time after the first time. */
    };
    Stage       stage;
    friend bool operator==(const UnitWorkloadStage &stage0, const UnitWorkloadStage &stage1)
    {
        return stage0.stage == stage1.stage;
    }
};
/** Type of memory used by a Workload Tensor
 *
 */
enum class MemoryType
{
    Core      = 0, /**< Core memory used by the Workload Tensor, e.g. for argument tensors */
    Auxiliary = 1, /**< Auxiliary memory required by the Workload Tensor, e.g. for temporary tensors */
};

using AuxMemoryLifetime = MemoryLifetime;

/** Memory Info for a @ref WorkloadTensor of Auxiliary memory type. This communicates to the user how much additional
 *  memory is required for auxiliary tensors
 */
struct AuxMemoryInfo
{
    AuxMemoryInfo() = default;

    AuxMemoryInfo(size_t size, size_t alignment = 0) noexcept
        : size(size),
          alignment(alignment)
    {
    }

    AuxMemoryInfo(AuxMemoryLifetime lifetime, size_t size, size_t alignment = 0) noexcept
        : lifetime(lifetime),
          size(size),
          alignment(alignment)
    {
    }
    friend bool operator==(const AuxMemoryInfo &info0, const AuxMemoryInfo &info1)
    {
        return info0.lifetime == info1.lifetime && info0.size == info1.size && info0.alignment == info1.alignment;
    }

    AuxMemoryLifetime lifetime{ AuxMemoryLifetime::Temporary }; /**< Memory lifetime*/
    size_t            size{ 0 };                                /**< Total memory size in bytes */
    size_t            alignment{ 64 };                          /**< Memory alignment in bytes */
};

/** A descriptor for IWorkload Tensors.
 */
struct WorkloadTensor
{
    using Id = DependencyGraph::Id;
    Id            id{};          /**< Id of the workload tensor */
    ITensorInfo *info{};         /**< TensorInfo associated with the workload tensor */
    MemoryType    memory_type{}; /**< Memory type */
    AuxMemoryInfo memory_info{}; /**< Auxiliary memory information. This can be ignored if the memory type is Core */
};
/** The basic atomic unit in an @ref IWorkload. It contains exactly one kernel to run.
 *
 */
struct UnitWorkload
{
    using Id = DependencyGraph::Id;
    Id                id{};    /**< Id of the unit workload */
    UnitWorkloadStage stage{}; /**< Stage */
};

/** Run-time-agnostic, platform-specific graph that describes everything required to run a workload
 *  It can be configured into an Arm Compute Library runtime, integrated into the runtime of another framework, or integrated into the compilation flow
 */
struct IWorkload
{
    using UnitWorkId     = UnitWorkload::Id;
    using Tid            = WorkloadTensor::Id;
    IWorkload()          = default;
    virtual ~IWorkload() = default;
    DependencyGraph graph{}; /**< Dependency graph of the workload tensors and the unit workloads */
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IWORKLOAD_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */