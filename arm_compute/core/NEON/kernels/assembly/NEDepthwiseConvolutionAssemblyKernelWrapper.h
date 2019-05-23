/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_ASSEMBLY_DEPTHWISE_CONVOLUTION_ASSEMBLY_WRAPPER_KERNEL_H__
#define __ARM_COMPUTE_ASSEMBLY_DEPTHWISE_CONVOLUTION_ASSEMBLY_WRAPPER_KERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include "arm_compute/core/NEON/kernels/convolution/depthwise/depthwise.hpp"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** This class is a wrapper for the depthwise convolution assembly kernels.  */
class NEDepthwiseConvolutionAssemblyKernelWrapper final : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDepthwiseConvolutionAssemblyKernelWrapper";
    }

    /** Default constructor */
    NEDepthwiseConvolutionAssemblyKernelWrapper()
        : _kernel(nullptr)
    {
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionAssemblyKernelWrapper(const NEDepthwiseConvolutionAssemblyKernelWrapper &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionAssemblyKernelWrapper &operator=(const NEDepthwiseConvolutionAssemblyKernelWrapper &) = delete;
    /** Default Move Constructor. */
    NEDepthwiseConvolutionAssemblyKernelWrapper(NEDepthwiseConvolutionAssemblyKernelWrapper &&) = default;
    /** Default move assignment operator */
    NEDepthwiseConvolutionAssemblyKernelWrapper &operator=(NEDepthwiseConvolutionAssemblyKernelWrapper &&) = default;

    /** Initialise the kernel's input and output.
     *
     * @param[in] kernel Pointer to an assembly kernel implementation.
     */
    void configure(depthwise::IDepthwiseConvolution *kernel)
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(kernel)));
        _kernel = kernel;
        Window win;
        win.set(Window::DimX, Window::Dimension(0, _kernel->get_window(), 1));
        INEKernel::configure(win);
    }

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override
    {
        ARM_COMPUTE_ERROR_ON_NULLPTR((reinterpret_cast<void *>(_kernel)));
        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
        auto first = window.x().start();
        auto last  = window.x().end();
        _kernel->run(first, last, info.thread_id);
    }

private:
    depthwise::IDepthwiseConvolution *_kernel;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_ASSEMBLY_DEPTHWISE_CONVOLUTION_ASSEMBLY_WRAPPER_KERNEL_H__ */
