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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTCAST
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTCAST

#include "arm_compute/dynamic_fusion/sketch/attributes/CastAttributes.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"

namespace arm_compute
{
/** Forward declaration */
class ITensorInfo;
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
template <typename T>
class ArgumentPack;

/** Component specific settings
 */
class ClComponentCastSettings
{
public:
private:
};

/** Forward declaration */
class ClTemplateCast;

class ClComponentCast final : public IGpuKernelComponent
{
public:
    /** Attributes are a set of backend-agnostic parameters that define what a component does */
    using Attributes = CastAttributes;
    /** Settings are a set of backend-specific parameters that influence the implementation of a component */
    using Settings = ClComponentCastSettings;

    /** Validate the component
     *
     * @param[in]     properties Component properties @ref Properties
     * @param[in,out] tensors    Tensor arguments to the component
     * @param[in]     attributes Component attributes @ref Attributes
     * @param[in]     settings   Component settings @ref Settings
     *
     * @return Status        Validation results
     *
     * Tensor argument names:
     * - ACL_SRC_0: Input
     * - ACL_DST_0: Output
     *
     * Tensor argument constness:
     * - ACL_SRC_0: Const
     * - ACL_DST_0: Const
     *
     * Valid data layouts:
     * - All
     *
     ** Valid data type configurations:
     * |ACL_SRC_0      |ACL_DST_0                              |
     * |:--------------|:--------------------------------------|
     * |U8             | S8, U16, S16, U32, S32, F16, F32      |
     * |U16            | U8, S8, S16, U32, S32, F16, F32       |
     * |S16            | U8, S8, U16, U32, S32, F16, F32       |
     * |U32            | U8, S8, U16, S16, S32, F16, F32       |
     * |S32            | U8, S8, U16, S16, U32, F16, F32       |
     * |F16            | U8, S8, U16, S16, U32, S32, F32       |
     * |F32            | U8, S8, U16, S16, U32, S32, F16       |
     */
    static Status validate(
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors,
        const Attributes                &attributes,
        const Settings                  &settings);

    /** Constructor
     *
     * Similar to @ref ClComponentCast::validate()
     */
    ClComponentCast(ComponentId                      id,
                    const Properties                &properties,
                    const ArgumentPack<ITensorInfo> &tensors,
                    const Attributes                &attributes,
                    const Settings                  &settings);

    /** Destructor */
    ~ClComponentCast() override;
    /** Prevent instances of this class from being copy constructed */
    ClComponentCast(const ClComponentCast &component) = delete;
    /** Prevent instances of this class from being copied */
    ClComponentCast &operator=(const ClComponentCast &component) = delete;
    /** Allow instances of this class to be move constructed */
    ClComponentCast(ClComponentCast &&component) = default;
    /** Allow instances of this class to be moved */
    ClComponentCast &operator=(ClComponentCast &&component) = default;
    /** Get template writer for the component */
    const IGpuTemplateComponentWriter *template_writer() const override;
    /** Get component type */
    GpuComponentType type() const override
    {
        return GpuComponentType::Simple;
    }

private:
    std::unique_ptr<ClTemplateCast> _component_writer;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTCAST */
