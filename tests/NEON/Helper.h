/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_NEON_HELPER_H__
#define __ARM_COMPUTE_TEST_NEON_HELPER_H__

#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/NEON/INESimpleFunction.h"
#include "support/ToolchainSupport.h"
#include "tests/Globals.h"

#include <algorithm>
#include <array>
#include <vector>

namespace arm_compute
{
namespace test
{
template <typename D, typename T, typename... Ts>
void fill_tensors(D &&dist, std::initializer_list<int> seeds, T &&tensor, Ts &&... other_tensors)
{
    const std::array < T, 1 + sizeof...(Ts) > tensors{ { std::forward<T>(tensor), std::forward<Ts>(other_tensors)... } };
    std::vector<int> vs(seeds);
    ARM_COMPUTE_ERROR_ON(vs.size() != tensors.size());
    int k = 0;
    for(auto tp : tensors)
    {
        library->fill(Accessor(*tp), std::forward<D>(dist), vs[k++]);
    }
}

/** This template synthetizes an INESimpleFunction which runs the given kernel K */
template <typename K>
class NESynthetizeFunction : public INESimpleFunction
{
public:
    /** Configure the kernel.
     *
     * @param[in] args Configuration arguments.
     */
    template <typename... Args>
    void configure(Args &&... args)
    {
        auto k = arm_compute::support::cpp14::make_unique<K>();
        k->configure(std::forward<Args>(args)...);
        _kernel = std::move(k);
    }
};

/** As above but this also setups a Zero border on the input tensor of the specified bordersize */
template <typename K, int bordersize>
class NESynthetizeFunctionWithZeroConstantBorder : public INESimpleFunction
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
        auto k = arm_compute::support::cpp14::make_unique<K>();
        k->configure(first, std::forward<Args>(args)...);
        _kernel = std::move(k);
        _border_handler.configure(first, BorderSize(bordersize), BorderMode::CONSTANT, PixelValue());
    }
};

/** As above but this also setups a Zero border on the input tensor of the kernel's bordersize */
template <typename K>
class NESynthetizeFunctionWithZeroConstantKernelBorder : public INESimpleFunction
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
        auto k = arm_compute::support::cpp14::make_unique<K>();
        k->configure(first, std::forward<Args>(args)...);
        _kernel = std::move(k);
        _border_handler.configure(first, BorderSize(_kernel->border_size()), BorderMode::CONSTANT, PixelValue());
    }
};

} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_NEON_HELPER_H__ */
