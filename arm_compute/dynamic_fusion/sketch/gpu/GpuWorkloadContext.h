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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXT
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXT

#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"

#include <memory>

namespace arm_compute
{
/** Forward declaration */
class CLCompileContext;
namespace experimental
{
namespace dynamic_fusion
{
/** Gpu Information such as the Gpu target (for example, G76) */
using GpuTarget = ::arm_compute::GPUTarget;

/** Gpu Language */
enum class GpuLanguage
{
    OpenCL,
    Unknown
};
/** Provide context necessary for the creation and configuration of a workload
 * e.g. gpu targets and capabilities, cl::Device for querying OpenCl extensions. Both can affect how a kernel is generated
 *
 * This context is shared between different operators within a sketch, and has to stay valid for the entire workload creation session.
 * This context may also be shared between different sketches.
 *
 * This class only contains information for workload creation, but not for runtime (e.g. cl::Queue for enqueueing the kernels)
 */
class GpuWorkloadContext
{
public:
    class Impl;

    /** Constructor */
    GpuWorkloadContext(CLCompileContext *cl_compile_context);
    /** Destructor */
    ~GpuWorkloadContext();
    /** Prohibit instances of this class to be copy constructed */
    GpuWorkloadContext(const GpuWorkloadContext &config) = delete;
    /** Prohibit instances of this class to be copied */
    GpuWorkloadContext &operator=(const GpuWorkloadContext &config) = delete;
    /** Allow instances of this class to be move constructed */
    GpuWorkloadContext(GpuWorkloadContext &&config);
    /** Allow instances of this class to be moved */
    GpuWorkloadContext &operator=(GpuWorkloadContext &&config);
    /** Get @ref GpuLanguage of the context */
    GpuLanguage gpu_language() const;
    /** Get @ref GpuTarget of the context */
    GpuTarget gpu_target() const;
    /** Get @ref CLCompileContext
     * If the gpu language is not OpenCL, then return nullptr
     */
    const CLCompileContext *cl_compile_context() const;

    /** Create a @ref TensorInfo associated with the workload context.
     *
     * @return TensorInfo Newly created tensor info
     */
    template <typename... TArgs>
    TensorInfo create_tensor_info(TArgs &&... args)
    {
        auto tensor_info = TensorInfo(std::forward<TArgs>(args)...);
        register_user_tensor(tensor_info);
        return tensor_info;
    }

    /** Get the internal implementation */
    Impl &implementation();

    /** Get the internal implementation */
    const Impl &implementation() const;

private:
    /** Set a new ID to the tensor info and register its memory descriptor to the context.
     *
     * @param[in,out] tensor_info @ref ITensorInfo to be registered.
     */
    void register_user_tensor(ITensorInfo &tensor_info);

    /** Internal implementation */
    std::unique_ptr<Impl> _impl;
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXT */
