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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATEACTIVATION
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATEACTIVATION

#include "arm_compute/core/experimental/Types.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentActivation.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/GpuKernelVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/IGpuTemplateComponentWriter.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class ClTemplateActivation final : public IGpuTemplateComponentWriter
{
public:
    using Attributes = ClComponentActivation::Attributes;

    /** Constructor
     *
     * @param[in] id         Component id
     * @param[in] tensors    Tensor arguments to the components
     * @param[in] attributes Component attributes
     */
    ClTemplateActivation(ComponentId id, const ArgumentPack<ITensorInfo> &tensors, const Attributes &attributes);

    /** Destructor */
    ~ClTemplateActivation() override = default;

    /** Prevent instances of this class from being copy constructed */
    ClTemplateActivation(const ClTemplateActivation &activation) = delete;

    /** Prevent instances of this class from being copied */
    ClTemplateActivation &operator=(const ClTemplateActivation &activation) = delete;

    /** Allow instances of this class to be move constructed */
    ClTemplateActivation(ClTemplateActivation &&activation) = default;

    /** Allow instances of this class to be moved */
    ClTemplateActivation &operator=(ClTemplateActivation &&activation) = default;

    /** Generate kernel component name */
    std::string get_name() const override;

    /** Generate kernel component code template
     *
     * @param[in] comp_group Component group of which the component is a part of
     *
     * @return std::string Component code
     */
    std::string get_component_code(const ComponentGroup &comp_group) const override;

    /** Declare all variables used by the component in the @p vtable
     *
     * @param[out] vtable     Variable table
     * @param[in]  comp_group Component group of which the component is a part of
     */
    void declare_variables(GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const override;

    /** Generate the tag look-up table used to instantiate the component code.
     *
     * @param[in] vtable     Variable table
     * @param[in] comp_group Component group of which the component is a part of
     *
     * @return TagLUT  Tag lookup table
     */
    TagLUT get_tag_lut(const GpuKernelVariableTable &vtable, const ComponentGroup &comp_group) const override;

    /** Generate the build options used in the component
     *
     * @param[in] comp_group Component group of which the component is a part of
     *
     * @return CLBuildOptions Build options
     */
    CLBuildOptions get_build_options(const ComponentGroup &comp_group) const override;

    /** Generate the component config id string used for tuning */
    std::string get_config_id() const override;

    /** Generate the header list used in the component */
    std::set<std::string> get_headers_list() const override;

    /** Generate the execution window for the component */
    Window get_window() const override;

private:
    const ITensorInfo *_src;
    const ITensorInfo *_dst;
    Attributes         _attributes;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATEACTIVATION */
