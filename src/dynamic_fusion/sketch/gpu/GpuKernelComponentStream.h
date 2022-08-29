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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTSTREAM
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTSTREAM

#include "arm_compute/dynamic_fusion/sketch/MemoryDescriptor.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSourceCode.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class GpuComponentServices;
class IGpuKernelComponent;

/** A linear sequence of component groups serialized from the @ref GpuKernelComponentGraph
 *  Each component group in the stream denotes a complete kernel that may consist of multiple components
 *
 * The main purposes of this class are:
 *  - Facilitate component fusion algorithm by allowing insertions of new component groups into the stream
 *  - Invoke kernel writer and assemble the final @ref GpuWorkloadSourceCode
 */
class GpuKernelComponentStream
{
public:
    /** Constructor
     *
     * @param[in] services @ref GpuComponentServices to be used throughout the stream
     * @param[in] mem_map  @ref MemoryDescriptor map used to assemble the @ref GpuWorkloadSourceCode
     */
    GpuKernelComponentStream(GpuComponentServices *services, const MemoryDescriptorMap &mem_map);
    /** Allow instances of this class to be copy constructed */
    GpuKernelComponentStream(const GpuKernelComponentStream &stream) = default;
    /** Allow instances of this class to be copied */
    GpuKernelComponentStream &operator=(const GpuKernelComponentStream &stream) = default;
    /** Allow instances of this class to be move constructed */
    GpuKernelComponentStream(GpuKernelComponentStream &&stream) = default;
    /** Allow instances of this class to be moved */
    GpuKernelComponentStream &operator=(GpuKernelComponentStream &&stream) = default;
    /** Generate and assemble @ref GpuWorkloadSourceCode from the stream */
    GpuWorkloadSourceCode write_workload_code();
    /** Insert a new component group in the stream.
     * Subsequent components are added to this group until end of stream or the next new_component_group is called
     */
    void new_component_group();
    /** Add a component to the previously created component group
     *  Throw an error if no component group is present in the stream
     *
     * @param[in] component Component to be inserted
     *
     * @return true      If the operation is successful
     * @return false     Otherwise
     */
    bool add_component(IGpuKernelComponent *component);

private:
    GpuComponentServices                *_services;
    std::vector<GpuKernelComponentGroup> _component_groups{};
    MemoryDescriptorMap                  _mem_map{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTSTREAM */
