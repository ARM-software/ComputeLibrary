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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPULOGICALKERNEL
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPULOGICALKERNEL

#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelSourceCode.h"

#include <memory>
#include <vector>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
class GpuComponentServices;
class IGpuKernelComponent;

/** A wrapper-processor of a @ref GpuKernelComponentGroup
 * It adds the load (if any) and store components to the component group
 * The @ref GpuLogicalKernel represents a complete kernel, and can proceed to invoke any kernel writer to generate the full kernel code
 */
class GpuLogicalKernel
{
public:
    /** Constructor
     *
     * @param[in] services   @ref GpuComponentServices to be used
     * @param[in] components Component group from which this logical kernel is initialized
     */
    explicit GpuLogicalKernel(GpuComponentServices *services, const GpuKernelComponentGroup &components);
    /** Allow instances of this class to be copy constructed */
    GpuLogicalKernel(const GpuLogicalKernel &) = default;
    /** Allow instances of this class to be copied */
    GpuLogicalKernel &operator=(const GpuLogicalKernel &) = default;
    /** Allow instances of this class to be move constructed */
    GpuLogicalKernel(GpuLogicalKernel &&) = default;
    /** Allow instances of this class to be moved */
    GpuLogicalKernel &operator=(GpuLogicalKernel &&) = default;
    /** Generate a @ref GpuKernelSourceCode */
    GpuKernelSourceCode write_kernel_code();

private:
    GpuKernelComponentGroup                           _comp_group{};
    std::vector<std::unique_ptr<IGpuKernelComponent>> _store_components{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPULOGICALKERNEL */
