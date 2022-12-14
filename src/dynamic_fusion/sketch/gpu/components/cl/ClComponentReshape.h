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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTRESHAPE
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTRESHAPE

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
class ClTemplateReshape;

class ClComponentReshape final : public IGpuKernelComponent
{
public:
public:
    /** Validate the component
     *
     * @param[in,out] tensors Tensor arguments to the component
     *
     * @return Status       Validation results
     *
     * Tensor argument names:
     * - ACL_SRC_0: src
     * - ACL_DST_0: dst
     *
     * Tensor argument constness:
     * - ACL_SRC_0: Const
     * - ACL_DST_0: Const
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * - All
     */
    static Status validate(const ArgumentPack<ITensorInfo> &tensors);

    /** Constructor
     *
     * @param[in] id         Component id
     * @param[in] properties Component properties @ref Properties
     * @param[in] tensors    Tensor arguments to the component
     */
    ClComponentReshape(
        ComponentId                      id,
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors);

    /** Destructor */
    ~ClComponentReshape() override;
    /** Prevent instances of this class from being copy constructed */
    ClComponentReshape(const ClComponentReshape &component) = delete;
    /** Prevent instances of this class from being copied */
    ClComponentReshape &operator=(const ClComponentReshape &component) = delete;
    /** Allow instances of this class to be move constructed */
    ClComponentReshape(ClComponentReshape &&component) = default;
    /** Allow instances of this class to be moved */
    ClComponentReshape &operator=(ClComponentReshape &&component) = default;
    /** Get template writer for the component */
    const IGpuTemplateComponentWriter *template_writer() const override;
    /** Get component type */
    GpuComponentType type() const override
    {
        return GpuComponentType::Complex;
    }

private:
    std::unique_ptr<ClTemplateReshape> _component_writer;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTRESHAPE */
