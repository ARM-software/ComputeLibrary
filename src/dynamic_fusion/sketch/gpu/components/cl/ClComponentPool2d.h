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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTPOOL2D
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTPOOL2D

#include "arm_compute/dynamic_fusion/sketch/gpu/operators/GpuPool2d.h"
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
class Pool2dAttributes;

/** Forward declaration */
class ClTemplatePool2d;

class ClComponentPool2d final : public IGpuKernelComponent
{
public:
    /** Attributes are a set of backend-agnostic parameters that define what a component does */
    using Attributes = Pool2dAttributes;
    /** Settings are a set of backend-specific parameters that influence the implementation of a component */
    using Settings = GpuPool2dSettings;

public:
    /** Validate the component
     *
     * @param[in]     properties Component properties
     * @param[in,out] tensors    Tensor arguments to the component
     * @param[in]     attributes Component attributes
     * @param[in]     settings   Component settings
     *
     * @return Status       Validation results
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
     * - NHWC
     *
     * Valid data type configurations:
     * |ACL_SRC_0      |ACL_DST_0      |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     */
    static Status validate(
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors,
        const Attributes                &attributes,
        const Settings                  &settings);

    /** Constructor
     *
     * @param[in]     id         Unique Component Identifier within a workload
     * @param[in]     properties Component properties
     * @param[in,out] tensors    Tensor arguments to the component
     * @param[in]     attributes Component attributes
     * @param[in]     settings   Component settings
     */
    ClComponentPool2d(
        ComponentId                      id,
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors,
        const Attributes                &attributes,
        const Settings                  &settings);

    /** Destructor */
    ~ClComponentPool2d() override;

    /** Prevent instances of this class from being copy constructed */
    ClComponentPool2d(const ClComponentPool2d &component) = delete;

    /** Prevent instances of this class from being copied */
    ClComponentPool2d &operator=(const ClComponentPool2d &component) = delete;

    /** Allow instances of this class to be move constructed */
    ClComponentPool2d(ClComponentPool2d &&component) = default;

    /** Allow instances of this class to be moved */
    ClComponentPool2d &operator=(ClComponentPool2d &&component) = default;

    /** Get template writer for the component */
    const IGpuTemplateComponentWriter *template_writer() const override;

    /** Get component type */
    GpuComponentType type() const override
    {
        return GpuComponentType::Unfusable;
    }

private:
    std::unique_ptr<ClTemplatePool2d> _component_writer;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTPOOL2D */
