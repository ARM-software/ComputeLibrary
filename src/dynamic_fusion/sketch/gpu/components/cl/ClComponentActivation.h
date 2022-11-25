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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTACTIVATION
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTACTIVATION

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

/** Forward declaration */
class ClTemplateActivation;

class ClComponentActivation final : public IGpuKernelComponent
{
public:
    /** Attributes are a set of backend-agnostic parameters that define what a component does */
    using Attributes = ActivationLayerInfo;

    /** Validate the component
     *
     * @param[in]      properties Component properties @ref Properties
     * @param[in, out] tensors    Tensor arguments to the component
     * @param[in]      attributes Component attributes @ref Attributes
     *
     * @return Status       Validation results
     *
     * Tensor argument names:
     * - ACL_SRC: Input
     * - ACL_DST: Output
     *
     * Tensor argument constness:
     * - ACL_SRC: Const
     * - ACL_DST: Const
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |ACL_SRC        |ACL_DST        |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     */
    static Status validate(
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors,
        const Attributes                &attributes);

    /** Constructor
     *
     * Similar to @ref ClComponentActivation::validate()
     */
    ClComponentActivation(
        ComponentId                      id,
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors,
        const Attributes                &attributes);

    /** Destructor */
    ~ClComponentActivation() override = default;

    /** Prevent instances of this class from being copy constructed */
    ClComponentActivation(const ClComponentActivation &component) = delete;

    /** Prevent instances of this class from being copied */
    ClComponentActivation &operator=(const ClComponentActivation &component) = delete;

    /** Allow instances of this class to be move constructed */
    ClComponentActivation(ClComponentActivation &&component) = default;

    /** Allow instances of this class to be moved */
    ClComponentActivation &operator=(ClComponentActivation &&component) = default;

    /** Get template writer for the component */
    const IGpuTemplateComponentWriter *template_writer() const override;

    /** Get component type */
    GpuComponentType type() const override
    {
        return GpuComponentType::Simple;
    }

private:
    std::unique_ptr<ClTemplateActivation> _component_writer;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTACTIVATION */
