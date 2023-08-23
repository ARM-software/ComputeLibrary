/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_IGPUCKWCOMPONENTDRIVER
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_IGPUCKWCOMPONENTDRIVER

#include "arm_compute/core/Window.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/components/Types.h"

namespace arm_compute
{
class ITensorInfo;
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
class GpuKernelComponentGroup;
class GpuCkwVariableTable;
class GpuCkwScopedKernelWriter;

/** An interface used by @ref GpuCkwDriver to write source code for a kernel component
 *
 * There are 3 main architecture layers for using Compute Kernel Writer (Ckw) inside ACL's dynamic fusion module
 * From top level to bottom level:
 * | Layer          | Library
 * ===========================
 * | dynamic_fusion |   acl
 * | ckw_driver     |   acl
 * | ckw            |   ckw
 *
 * ckw_driver is a glue layer that directs how fused code is produced using the ckw library
 *
 * There are two main groups within ckw_driver:
 * - @ref GpuCkwDriver is a global driver that coordinates how the final fused code along with all the info necessary
 *   for run time execution is produced using ckw
 * - Various classes implementing @ref IGpuCkwComponentDriver is a component driver that directs ckw to generate kernel component code (e.g. activation, store etc.)
 *
 * The overall flow goes like this:
 * In dynamic_fusion module, @ref GpuLogicalKernel instantiates a @ref GpuCkwDriver from a @ref GpuKernelComponentGroup
 * The logical kernel then uses the global driver's various interfaces to generate the code info.
 * In particular, the @ref GpuCkwDriver::get_code() interface will call into each @ref IGpuCkwComponentDriver::write_component_code()
 */
class IGpuCkwComponentDriver
{
public:
    using ComponentGroup = GpuKernelComponentGroup;

public:
    /** Constructor
     *
     * @param[in] id      Component id
     * @param[in] tensors Tensor arguments to the components
     */
    IGpuCkwComponentDriver(ComponentId id, const ArgumentPack<ITensorInfo> &tensors)
        : _id{ id }, _tensors{ tensors }
    {
    }
    /** Destructor */
    virtual ~IGpuCkwComponentDriver()
    {
    }
    /** Generate kernel component code
     *
     * @param[in]      comp_group Component group of which the component is a part of
     * @param[in, out] vtable     Table of variables declared by each component
     * @param[in, out] writer     CKW writer that writes code scoped to this kernel component.
     *
     *                            @note @p writer can only be passed via value since the new scope is created in the copy constructor
     */
    virtual void write_component_code(const ComponentGroup &comp_group, GpuCkwVariableTable &vtable, GpuCkwScopedKernelWriter writer) const = 0;
    /** Get tensor arguments */
    ArgumentPack<ITensorInfo> tensors() const
    {
        return _tensors;
    }
    /** Generate the execution window for the component */
    virtual Window get_window() const
    {
        return Window{};
    }
    /** Generate the name of the component
     *
     * This will be concatenated with other components' names to form the name of the kernel
     */
    virtual std::string get_name(const ComponentGroup &comp_group) const
    {
        ARM_COMPUTE_UNUSED(comp_group);
        return "unnamed";
    }
    /** Generate the tuner id of the component
     *  This id should capture all the parameters that distinguish one kernel's lws tuning from another.
     *  e.g. two components that are identical in every other way, but have output tensor dimensions should
     *  have different tuner ids, because the lws of one may not be optimal on the other.
     *
     * This will be concatenated with other components' tuner id to form the tuner id of the kernel
     */
    virtual std::string get_tuner_id(const ComponentGroup &comp_group) const
    {
        ARM_COMPUTE_UNUSED(comp_group);
        return "";
    }
    /** Get component id */
    ComponentId id() const
    {
        return _id;
    }

private:
    ComponentId               _id{ -1 };
    ArgumentPack<ITensorInfo> _tensors{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_IGPUCKWCOMPONENTDRIVER */
