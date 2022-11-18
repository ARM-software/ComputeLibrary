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
#ifndef SRC_DYNAMIC_FUSION_RUNTIME_GPU_CL_CLKERNELRUNTIME
#define SRC_DYNAMIC_FUSION_RUNTIME_GPU_CL_CLKERNELRUNTIME

#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelSourceCode.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
class GpuKernelSourceCode;

/** OpenCL runtime to run a single kernel */
class ClKernelRuntime final : public opencl::IClKernel
{
public:
    /** Configure the kernel runtime
     *
     * @param[in] compile_ctx OpenCL compile context
     * @param[in] code        Kernel source code
     */
    void configure(const opencl::ClCompileContext &compile_ctx, const GpuKernelSourceCode &code);
    /** Run the kernel
     *
     * @param[in,out] tensors @ref ITensorPack object containing run-time tensor memories
     * @param[in]     window  Execution window
     * @param[in]     queue   OpenCL command queue
     */
    virtual void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

private:
    /** Set a kernel tensor argument
     *
     * @param[in,out] idx       Index at which to start adding the tensor's arguments. Will be incremented by the number of kernel arguments set.
     * @param[in]     arg       Kernel argument descriptor accompanying @p tensor
     * @param[in]     tensor    Tensor to set as an argument of the object's kernel
     * @param[in]     arg_slice Window the kernel will be run on
     * @param[out]    cl_images Extra cl images created from the tensor (will need to be retained until the kernel is enqueued)
     */
    inline void add_tensor_argument(unsigned int &idx, const GpuKernelArgumentInfo &arg, const ICLTensor *tensor, const Window &arg_slice, std::vector<cl::Image2D> &cl_images);

private:
    GpuKernelArgumentList _arguments{}; /** All kernel arguments required by the runtime */
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_RUNTIME_GPU_CL_CLKERNELRUNTIME */
