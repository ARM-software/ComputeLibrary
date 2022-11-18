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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_GPUKERNELCOMPONENTFACTORY
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_GPUKERNELCOMPONENTFACTORY

#include "Types.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"
#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Factory class that creates new instances of @ref IGpuKernelComponent by assigning new component ids
 */
class GpuKernelComponentFactory
{
public:
    /** Create a new kernel component
     *
     * @tparam T      Any polymorphic type descending from @ref IGpuKernelComponent
     * @tparam Args   Argument types to construct the kernel component
     *
     * @param[in] args Arguments to construct the kernel component
     *
     * @return std::unique_ptr<IGpuKernelComponent>
     */
    template <typename T, typename... Args>
    std::unique_ptr<IGpuKernelComponent> create(Args &&... args)
    {
        return std::make_unique<T>(_count++, std::forward<Args>(args)...);
    }

private:
    ComponentId _count{ 0 };
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_GPUKERNELCOMPONENTFACTORY */
