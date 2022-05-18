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
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLWORKLOAD_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLWORKLOAD_H

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Window.h"

#include "arm_compute/core/experimental/IWorkload.h"
#include "arm_compute/core/experimental/OperatorGraph.h"

#include <map>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Verbose and explicit way to enumerate all the tensor arguments variants used by
 *  all kernel implementations. This avoids any ambiguity in what kernel arguments are passed
 */
enum class ClKernelTensorArgType : int
{
    Scalar,

    Vector,

    Image,
    Image_Reinterpret_As_3D,
    Image_Export_To_ClImage2D,

    Image_3D, // 3D Tensor represented as a 2D Image + stride_z
    Image_3D_Export_To_ClImage2D,

    Tensor_3D,
    Tensor_4D,
    Tensor_4D_t_Buffer,
    Tensor_4D_t_Image
};

/** Describes all the info required to add a kernel argument at run time
 *
 *  @note This struct can later be expanded into a more concise and formal way to specify how to set up
 *  arguments for a kernel inside a @ref ClUnitWorkload
 */
struct ClKernelArgDescriptor
{
    ClKernelArgDescriptor() = default;
    ClKernelArgDescriptor(int arg_id, ClKernelTensorArgType type, bool slide_along_dimz = true)
        : arg_id{ arg_id }, tensor_arg_type{ type }, slide_along_dimz{ slide_along_dimz }
    {
    }
    ~ClKernelArgDescriptor() = default;
    friend bool operator==(const ClKernelArgDescriptor &arg0, const ClKernelArgDescriptor &arg1)
    {
        return (arg0.tensor_arg_type == arg1.tensor_arg_type) && (arg0.slide_along_dimz == arg1.slide_along_dimz);
    }
    int                   arg_id{ -1 };                                    /**< Arg ID in the blueprint, -1 means empty / uninitialized */
    ClKernelTensorArgType tensor_arg_type{ ClKernelTensorArgType::Image }; /**< tensor argument type */
    bool                  slide_along_dimz{ true };                        /**< @note slide_along_dimz will be moved out of this descriptor in later iterations */
};

using ClKernelArgList = std::map<int, ClKernelArgDescriptor>;

/** Descriptor containing information required to run a single ClWorkload
 */
struct ClExecutionDescriptor
{
    cl::NDRange suggested_lws{};              /**< Suggested local work-group size for optimal performance if not zero */
    cl::NDRange gws{};                        /**< Global work-group to be used */
    bool        skip_sliding_window{ false }; /**< Skip sliding window slices during execution loop */
};

/** Contains kernel code to be compiled and run in a ClUnitWorkload
 */
struct ClKernelCode
{
    friend bool operator==(const ClKernelCode &code0, const ClKernelCode &code1)
    {
        return (code0.name == code1.name) && (code0.code == code1.code) && (code0.config_id == code1.config_id) && (code0.build_options == code1.build_options) && (code0.window == code1.window)
               && (code0.arguments == code1.arguments);
    }
    std::string     name{};          /**< Kernel name */
    std::string     code{};          /**< Kernel source code */
    std::string     config_id{};     /**< Generated from blueprint based on complex component */
    CLBuildOptions  build_options{}; /**< Kernel build options */
    Window          window{};        /**< Execution window */
    ClKernelArgList arguments{};     /**< Kernel argument descriptors. map key is kernel ArgumentID */
};

/** A descriptor of ClWorkload Tensors.
 */
struct ClWorkloadTensor : public WorkloadTensor
{
    ClWorkloadTensor() = default;
    ClWorkloadTensor(Id id, ITensorInfo *info, MemoryType memory_type, const AuxMemoryInfo &memory_info, const ClKernelArgDescriptor &kernel_arg)
        : WorkloadTensor{ id, info, memory_type, memory_info }, kernel_arg{ kernel_arg }
    {
    }
    ClKernelArgDescriptor kernel_arg{};
    friend bool operator==(const ClWorkloadTensor &t0, const ClWorkloadTensor &t1)
    {
        return t0.info == t1.info && t0.memory_info == t1.memory_info && t0.memory_type == t1.memory_type && t0.kernel_arg == t1.kernel_arg;
    }
};

/** The basic atomic unit in a @ref ClWorkload. It contains exactly one kernel to run.
 */
struct ClUnitWorkload : public UnitWorkload
{
    ClUnitWorkload() = default;
    ClUnitWorkload(Id id, UnitWorkloadStage stage, const ClKernelCode &code)
        : UnitWorkload{ id, stage }, code{ code }
    {
    }
    friend bool operator==(const ClUnitWorkload &uworkload0, const ClUnitWorkload &uworkload1)
    {
        return uworkload0.stage == uworkload1.stage && uworkload0.code == uworkload1.code;
    }
    ClKernelCode code{};
};

/** GPU information for @ref ClWorkloadContext
 */
struct GpuInfo
{
    friend bool operator==(const GpuInfo &info0, const GpuInfo &info1)
    {
        return info0.target == info1.target;
    }
    GPUTarget target{ GPUTarget::UNKNOWN };
};

/** Context (device capabilities, platform details) associated with a ClWorkload
 *
 * It is required for building the @ref ClKernelCode and could also be used by the runtime (e.g. schedulers)
 */
struct ClWorkloadContext
{
    friend bool operator==(const ClWorkloadContext &ctx0, const ClWorkloadContext &ctx1)
    {
        return ctx0.gpu_info == ctx1.gpu_info;
    }
    GpuInfo gpu_info{};
};

/** Workload for Cl backend
 */
struct ClWorkload : public IWorkload
{
    Tid add_workload_tensor(ITensorInfo *info, MemoryType memory_type, const AuxMemoryInfo &memory_info, const ClKernelArgDescriptor &kernel_arg, Tid merge_point)
    {
        Tid id = graph.add_tensor(merge_point);
        if(tensors.find(id) == tensors.end())
        {
            tensors[id] = ClWorkloadTensor(id, info, memory_type, memory_info, kernel_arg);
        }
        return id;
    }
    UnitWorkId add_unit_workload(UnitWorkloadStage stage, const ClKernelCode &code, const std::vector<Tid> &inputs, const std::vector<Tid> &outputs)
    {
        auto op            = graph.add_operator(inputs, outputs);
        auto id            = op.second;
        unit_workloads[id] = ClUnitWorkload(id, stage, code);
        return id;
    }
    friend bool operator==(const ClWorkload &workload0, const ClWorkload &workload1)
    {
        return std::make_tuple(
                   workload0.graph, workload0.context, workload0.unit_workloads, workload0.tensors, workload0.op_tensor_id_lut)
               == std::make_tuple(
                   workload1.graph, workload1.context, workload1.unit_workloads, workload1.tensors, workload1.op_tensor_id_lut);
    }
    ClWorkloadContext context{};                             /**< Workload context*/
    std::map<UnitWorkId, ClUnitWorkload> unit_workloads{};   /**< Unit workloads to run*/
    std::map<Tid, ClWorkloadTensor>      tensors{};          /**< Workload tensors*/
    std::map<Tid, OpTensor::Id>          op_tensor_id_lut{}; /**< Map from ClWorkloadTensor to SRC and DST Operator Tensors (no need to store "intermediate" Operator Tensors)*/
    Status status{};                                         /**< For compatibility with the IOperator validate method. Store if the workload is valid or not. */
};

/** Build a @ref ClWorkload from an @ref OperatorGraph.
 *
 * @param[out] workload
 * @param[in] op_graph
 * @param[in] ctx
 * @return Status
 */
Status build(ClWorkload &workload, const OperatorGraph &op_graph, const ClWorkloadContext &ctx);

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLWORKLOAD_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */