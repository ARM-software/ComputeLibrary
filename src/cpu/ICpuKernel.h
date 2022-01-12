/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_ICPUKERNEL_H
#define ARM_COMPUTE_ICPUKERNEL_H

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "src/cpu/kernels/CpuKernelSelectionTypes.h"

namespace arm_compute
{
namespace cpu
{
enum class KernelSelectionType
{
    Preferred, /**< Retrieve the best implementation available for the given Cpu ISA, ignoring the build flags */
    Supported  /**< Retrieve the best implementation available for the given Cpu ISA that is supported by the current build */
};

template <class Derived>
class ICpuKernel : public ICPPKernel
{
public:
    /** Micro-kernel selector
     *
     * @param[in] selector       Selection struct passed including information to help pick the appropriate micro-kernel
     * @param[in] selection_type (Optional) Decides whether to get the best implementation for the given hardware or for the given build
     *
     * @return A matching micro-kernel else nullptr
     */

    template <typename SelectorType>
    static const auto *get_implementation(const SelectorType &selector, KernelSelectionType selection_type = KernelSelectionType::Supported)
    {
        using kernel_type = typename std::remove_reference<decltype(Derived::get_available_kernels())>::type::value_type;

        for(const auto &uk : Derived::get_available_kernels())
        {
            if(uk.is_selected(selector) && (selection_type == KernelSelectionType::Preferred || uk.ukernel != nullptr))
            {
                return &uk;
            }
        }

        return static_cast<kernel_type *>(nullptr);
    }
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_ICPUKERNEL_H */
