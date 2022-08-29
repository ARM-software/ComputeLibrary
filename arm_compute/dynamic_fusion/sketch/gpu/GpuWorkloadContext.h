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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXT
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXT

#include "arm_compute/core/GPUTarget.h"

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
    /** Constructor */
    GpuWorkloadContext(CLCompileContext *cl_compile_context);
    /** Allow instances of this class to be copy constructed */
    GpuWorkloadContext(const GpuWorkloadContext &config) = default;
    /** Allow instances of this class to be copied */
    GpuWorkloadContext &operator=(const GpuWorkloadContext &config) = default;
    /** Allow instances of this class to be move constructed */
    GpuWorkloadContext(GpuWorkloadContext &&config) = default;
    /** Allow instances of this class to be moved */
    GpuWorkloadContext &operator=(GpuWorkloadContext &&config) = default;
    /** Get @ref GpuLanguage of the context */
    GpuLanguage gpu_language() const;
    /** Get @ref GpuTarget of the context */
    GpuTarget gpu_target() const;
    /** Get @ref CLCompileContext
     * If the gpu language is not OpenCL, then return nullptr
     */
    const CLCompileContext *cl_compile_context() const;

private:
    GpuLanguage       _gpu_language{ GpuLanguage::Unknown };
    CLCompileContext *_cl_compile_ctx{ nullptr };
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_GPUWORKLOADCONTEXT */
