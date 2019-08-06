/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_ASSEMBLY_GEMM_KERNEL_WRAPPER_KERNEL_H__
#define __ARM_COMPUTE_ASSEMBLY_GEMM_KERNEL_WRAPPER_KERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include "gemm_common.hpp"

namespace arm_compute
{
class ITensor;

/** This class is a wrapper for the assembly kernels.
  *
  * Some kernels were written in assembly and highly optimised for specific CPUs like A53 or A55.
  * This class works as a wrapper for these assembly kernels. The arm compute library creates an instance
  * of NEGEMMAssemblyWrapperKernel and other auxiliary data structures to execute a single assembly kernel
  * in the context of an NEFunctions.
  *
  * The type T is the type of the actual kernel implemented in assembly which is of type
  *         template<typename To, typename Tr> class GemmCommon
  *
  *
  */
template <typename TypeInput, typename TypeOutput>
class NEGEMMAssemblyWrapperKernel final : public INEKernel
{
public:
    /** Constructor
     */
    NEGEMMAssemblyWrapperKernel()
        : _kernel(nullptr), _name("NEGEMMAssemblyWrapperKernel")
    {
    }

    NEGEMMAssemblyWrapperKernel(NEGEMMAssemblyWrapperKernel &)  = delete;
    NEGEMMAssemblyWrapperKernel(NEGEMMAssemblyWrapperKernel &&) = default;
    NEGEMMAssemblyWrapperKernel &operator=(NEGEMMAssemblyWrapperKernel &) = delete;

    const char *name() const override
    {
        return _name.c_str();
    }
    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(_kernel)));
        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
        auto first = window.x().start();
        auto last  = window.x().end();
        _kernel->execute(first, last, info.thread_id);
    }
    /** Initialise the kernel's input and output.
     *
     * @param[in] kernel      Pointer to an assembly kernel implementation.
     * @param[in] num_threads Number of concurrent threads which will execute the kernel.
     */
    void configure(arm_gemm::GemmCommon<TypeInput, TypeOutput> *kernel, std::string kernel_name_tag)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(kernel)));
        _kernel          = kernel;
        auto   win_last  = _kernel->get_window_size();
        Window win;
        win.set(Window::DimX, Window::Dimension(0, win_last, 1));
        INEKernel::configure(win);

        if(!kernel_name_tag.empty())
        {
            _name += "/" + kernel_name_tag;
        }
    }

private:
    arm_gemm::GemmCommon<TypeInput, TypeOutput> *_kernel;
    std::string _name;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_ASSEMBLY_GEMM_KERNEL_WRAPPER_KERNEL_H__ */
