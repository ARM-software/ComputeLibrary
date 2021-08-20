/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_CL_HELPER_H
#define ARM_COMPUTE_TEST_CL_HELPER_H

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/CL/functions/CLFill.h"
#include "arm_compute/runtime/IFunction.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/gpu/cl/IClOperator.h"
#include "src/gpu/cl/operators/ClFill.h"

#include "src/core/CL/ICLKernel.h"
#include "support/Cast.h"

#include <memory>

namespace arm_compute
{
namespace test
{
/** This template synthetizes a simple IOperator which runs the given kernel K */
template <typename K>
class CLSynthetizeOperator : public opencl::IClOperator
{
public:
    /** Configure the kernel.
     *
     * @param[in] args Configuration arguments.
     */
    template <typename... Args>
    void configure(Args &&... args)
    {
        auto k = std::make_unique<K>();
        k->configure(CLKernelLibrary::get().get_compile_context(), std::forward<Args>(args)...);
        _kernel = std::move(k);
    }
    /** Configure the kernel setting the GPU target as well
     *
     * @param[in] gpu_target GPUTarget to set
     * @param[in] args       Configuration arguments.
     */
    template <typename... Args>
    void configure(GPUTarget gpu_target, Args &&... args)
    {
        auto k = std::make_unique<K>();
        k->set_target(gpu_target);
        k->configure(CLKernelLibrary::get().get_compile_context(), std::forward<Args>(args)...);
        _kernel = std::move(k);
    }
    /** Validate input arguments
     *
     * @param[in] args Configuration arguments.
     */
    template <typename... Args>
    static Status validate(Args &&... args)
    {
        return K::validate(std::forward<Args>(args)...);
    }
};

/** As above but this also initializes to zero the input tensor */
template <typename K, int bordersize>
class CLSynthetizeOperatorInitOutputWithZeroAndWithZeroConstantBorder : public opencl::IClOperator
{
public:
    /** Configure the kernel.
     *
     * @param[in] first  First input argument.
     * @param[in] second Second input argument.
     * @param[in] args   Rest of the configuration arguments.
     */
    template <typename T, typename... Args>
    void configure(T first, T second, Args &&... args)
    {
        auto cctx = CLKernelLibrary::get().get_compile_context();
        auto k    = std::make_unique<K>();
        k->set_target(CLScheduler::get().target());
        k->configure(cctx, first, second, std::forward<Args>(args)...);
        _kernel = std::move(k);
        _border_handler.configure(cctx, first, BorderSize(bordersize), BorderMode::CONSTANT, PixelValue());
        _fill.configure(cctx, second, PixelValue());
    }

    // Inherited method overridden:
    void run(ITensorPack &tensors) override final
    {
        ARM_COMPUTE_ERROR_ON_MSG(!_kernel, "The CL kernel or function isn't configured");

        ITensorPack fill_pack = { { ACL_SRC, tensors.get_tensor(TensorType::ACL_DST) } };
        _fill.run(fill_pack);
        CLScheduler::get().enqueue_op(_border_handler, tensors);
        CLScheduler::get().enqueue_op(*_kernel, tensors);
    }

private:
    opencl::ClFill             _fill{};           /**< Kernel to initialize the tensor */
    CLFillBorderKernel         _border_handler{}; /**< Kernel to handle  borders */
    std::unique_ptr<ICLKernel> _kernel{};         /**< Kernel to run */
};

/** This template synthetizes an ICLSimpleFunction which runs the given kernel K */
template <typename K>
class CLSynthetizeFunction : public ICLSimpleFunction
{
public:
    /** Configure the kernel.
     *
     * @param[in] args Configuration arguments.
     */
    template <typename... Args>
    void configure(Args &&... args)
    {
        auto k = std::make_unique<K>();
        k->configure(std::forward<Args>(args)...);
        _kernel = std::move(k);
    }
    /** Configure the kernel setting the GPU target as well
     *
     * @param[in] gpu_target GPUTarget to set
     * @param[in] args       Configuration arguments.
     */
    template <typename... Args>
    void configure(GPUTarget gpu_target, Args &&... args)
    {
        auto k = std::make_unique<K>();
        k->set_target(gpu_target);
        k->configure(std::forward<Args>(args)...);
        _kernel = std::move(k);
    }
    /** Validate input arguments
     *
     * @param[in] args Configuration arguments.
     */
    template <typename... Args>
    static Status validate(Args &&... args)
    {
        return K::validate(std::forward<Args>(args)...);
    }
};

/** As above but this also setups a Zero border on the input tensor of the specified bordersize */
template <typename K, int bordersize>
class CLSynthetizeFunctionWithZeroConstantBorder : public ICLSimpleFunction
{
public:
    /** Configure the kernel.
     *
     * @param[in] first First configuration argument.
     * @param[in] args  Rest of the configuration arguments.
     */
    template <typename T, typename... Args>
    void configure(T first, Args &&... args)
    {
        auto k = std::make_unique<K>();
        k->configure(first, std::forward<Args>(args)...);
        _kernel = std::move(k);
        _border_handler->configure(first, BorderSize(bordersize), BorderMode::CONSTANT, PixelValue());
    }
};

/** As above but this also initializes to zero the input tensor */
template <typename K, int bordersize>
class CLSynthetizeFunctionInitOutputWithZeroAndWithZeroConstantBorder : public IFunction
{
public:
    /** Configure the kernel.
     *
     * @param[in] first  First input argument.
     * @param[in] second Second input argument.
     * @param[in] args   Rest of the configuration arguments.
     */
    template <typename T, typename... Args>
    void configure(T first, T second, Args &&... args)
    {
        auto k = std::make_unique<K>();
        k->set_target(CLScheduler::get().target());
        k->configure(first, second, std::forward<Args>(args)...);
        _kernel = std::move(k);
        _border_handler.configure(first, BorderSize(bordersize), BorderMode::CONSTANT, PixelValue());
        _fill.configure(second, PixelValue());
    }

    // Inherited method overridden:
    void run() override final
    {
        ARM_COMPUTE_ERROR_ON_MSG(!_kernel, "The CL kernel or function isn't configured");

        _fill.run();
        CLScheduler::get().enqueue(_border_handler, false);
        CLScheduler::get().enqueue(*_kernel);
    }

private:
    CLFill                     _fill{};           /**< Kernel to initialize the tensor */
    CLFillBorderKernel         _border_handler{}; /**< Kernel to handle  borders */
    std::unique_ptr<ICLKernel> _kernel{};         /**< Kernel to run */
};

/** As above but this also setups a Zero border on the input tensor of the kernel's bordersize */
template <typename K>
class ClSynthetizeOperatorWithBorder : public opencl::IClOperator
{
public:
    /** Configure the kernel.
     *
     * @param[in] first First configuration argument.
     * @param[in] args  Rest of the configuration arguments.
     */
    template <typename T, typename... Args>
    void configure(T first, Args &&... args)
    {
        auto k = std::make_unique<K>();
        k->configure(CLKernelLibrary::get().get_compile_context(), first, std::forward<Args>(args)...);
        _kernel = std::move(k);

        auto b = std::make_unique<CLFillBorderKernel>();
        b->configure(CLKernelLibrary::get().get_compile_context(), first, BorderSize(_kernel->border_size()), BorderMode::CONSTANT, PixelValue());
        _border_handler = std::move(b);
    }

    void run(ITensorPack &tensors) override
    {
        CLScheduler::get().enqueue(*_border_handler);
        CLScheduler::get().enqueue_op(*_kernel, tensors);
    }

private:
    std::unique_ptr<ICLKernel> _border_handler{ nullptr }; /**< Kernel to handle  borders */
    std::unique_ptr<ICLKernel> _kernel{};                  /**< Kernel to run */
};
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CL_HELPER_H */
