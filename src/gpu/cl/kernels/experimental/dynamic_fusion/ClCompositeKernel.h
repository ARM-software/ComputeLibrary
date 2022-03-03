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
#if defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)

#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLCOMPOSITEKERNEL_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLCOMPOSITEKERNEL_H

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
struct TensorBinding
{
    TensorBinding(const std::map<ArgumentID, ICLTensor *> binding)
        : _binding{ binding }
    {
    }
    std::map<ArgumentID, ICLTensor *> _binding;
};
class ClCompositeKernel : public opencl::IClKernel
{
public:
    void configure(const opencl::ClCompileContext &, const ClKernelCode &);

    /** Run the composite kernel
     *
     * @param tensors   TensorBinding object containing run-time tensors information
     * @param window    Execution window
     * @param queue     OpenCL Command queue
     * @param exec_desc Descriptor containing execution information
     */
    virtual void run_composite_op(TensorBinding &tensors, const Window &window, cl::CommandQueue &queue, const ClExecutionDescriptor &exec_desc) override;

private:
    inline void add_tensor_argument(unsigned int &idx, const ClKernelArgRuntimeDescriptor &arg, ICLTensor *tensor, const Window &arg_slice);

private:
    ClKernelArgList _arguments{}; /** All kernel arguments required by runtime */
};

/** Argument Binding.
 * Tensor Arguments to ICLKernel run_op method need to be passed via an ITensorPack. So the bind_arguments is essentially a converter from TensorBinding to ITensorPack
 */
Status bind_arguments(ITensorPack &tensor_pack, const ClKernelCode &, const TensorBinding &);

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif // ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_CLCOMPOSITEKERNEL_H

#endif // defined(ENABLE_EXPERIMENTAL_DYNAMIC_FUSION)