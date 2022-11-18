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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATESTORE
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATESTORE

#include "arm_compute/core/experimental/Types.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/GpuKernelVariableTable.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/IGpuTemplateComponentWriter.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class ClTemplateStore final : public IGpuTemplateComponentWriter
{
public:
    /** Constructor
     *
     * @param[in] id      Component id
     * @param[in] tensors Tensor arguments to the components
     */
    ClTemplateStore(ComponentId id, const ArgumentPack<ITensorInfo> &tensors);
    /** Prevent instances of this class from being copy constructed */
    ClTemplateStore(const ClTemplateStore &store) = delete;
    /** Prevent instances of this class from being copied */
    ClTemplateStore &operator=(const ClTemplateStore &store) = delete;
    /** Allow instances of this class to be move constructed */
    ClTemplateStore(ClTemplateStore &&store) = default;
    /** Allow instances of this class to be moved */
    ClTemplateStore &operator=(ClTemplateStore &&store) = default;
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

private:
    const ITensorInfo *_src;
    const ITensorInfo *_dst;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_TEMPLATE_WRITER_CL_CLTEMPLATESTORE */
