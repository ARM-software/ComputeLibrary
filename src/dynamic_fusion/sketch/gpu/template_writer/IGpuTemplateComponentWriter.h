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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_IGPUTEMPLATECOMPONENTWRITER
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_IGPUTEMPLATECOMPONENTWRITER

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Window.h"
#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/components/Types.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/GpuKernelVariableTable.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
class GpuKernelComponentGroup;
class GpuKernelVariableTable;

/** An interface used by @ref ClTemplateWriter to write source code for a kernel component
 */
class IGpuTemplateComponentWriter
{
public:
    using ComponentGroup = GpuKernelComponentGroup;

    /**For now all kernel intermeditate/destination tensors are expected to be of type Tensor_4D_t_Buffer*/
    static constexpr GpuKernelArgumentInfo::Type common_tensor_type = GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer;

public:
    /** Constructor
     *
     * @param[in] id      Component id
     * @param[in] tensors Tensor arguments to the components
     */
    IGpuTemplateComponentWriter(ComponentId id, const ArgumentPack<ITensorInfo> &tensors)
        : _id{ id }, _tensors{ tensors }
    {
    }
    /** Destructor */
    virtual ~IGpuTemplateComponentWriter()
    {
    }
    /** Generate kernel component name */
    virtual std::string get_name() const = 0;
    /** Generate kernel component code template
     *
     * @param[in] comp_group Component group of which the component is a part of
     *
     * @return std::string Component code
     */
    virtual std::string get_component_code(const ComponentGroup &comp_group) const = 0;
    /** Declare all variables used by the component in the @p vtable
     *
     * @param[out] vtable     Variable table
     * @param[in]  comp_group Component group of which the component is a part of
     */
    virtual void declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const = 0;
    /** Generate the tag look-up table used to instantiate the component code.
     *
     * @param[in] vtable     Variable table
     * @param[in] comp_group Component group of which the component is a part of
     *
     * @return TagLUT  Tag lookup table
     */
    virtual TagLUT get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const = 0;
    /** Generate additional macros used in the component */
    virtual std::string get_additional_macros() const
    {
        return "";
    }
    /** Generate the build options used in the component
     *
     * @param[in] comp_group Component group of which the component is a part of
     *
     * @return CLBuildOptions Build options
     */
    virtual CLBuildOptions get_build_options(const ComponentGroup &comp_group) const
    {
        ARM_COMPUTE_UNUSED(comp_group);
        return CLBuildOptions{};
    }
    /** Generate the component config id string used for tuning */
    virtual std::string get_config_id() const
    {
        return "";
    }
    /** Generate the header list used in the component */
    virtual std::set<std::string> get_headers_list() const
    {
        return std::set<std::string> {};
    }
    /** Generate the execution window for the component */
    virtual Window get_window() const
    {
        return Window{};
    }
    /** Get tensor arguments */
    ArgumentPack<ITensorInfo> tensors() const
    {
        return _tensors;
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
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_IGPUTEMPLATECOMPONENTWRITER */
