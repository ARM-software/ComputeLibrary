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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTSTORE
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTSTORE

#include "arm_compute/core/Error.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateStore.h"
#include <memory>

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

class ClComponentStore final : public IGpuKernelComponent
{
public:
    /** Validate the component
     *
     * @param[in] properties Component properties
     * @param[in] tensors    Tensor arguments to the components
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
     * - NHWC
     *
     * Valid data type configurations:
     * |ACL_SRC_0      |ACL_DST_0      |
     * |:--------------|:--------------|
     * |All            |All            |
     */
    static Status validate(
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors);
    /** Constructor
     *
     * Similar to @ref ClComponentStore::validate()
     */
    ClComponentStore(ComponentId id, const Properties &properties, const ArgumentPack<ITensorInfo> &tensors);
    /** Destructor */
    ~ClComponentStore() override;
    /** Prevent instances of this class from being copy constructed */
    ClComponentStore(const ClComponentStore &component) = delete;
    /** Prevent instances of this class from being copied */
    ClComponentStore &operator=(const ClComponentStore &component) = delete;
    /** Allow instances of this class to be move constructed */
    ClComponentStore(ClComponentStore &&component) = default;
    /** Allow instances of this class to be moved */
    ClComponentStore &operator=(ClComponentStore &&component) = default;
    /** Get template writer for the component */
    const IGpuTemplateComponentWriter *template_writer() const override;
    /** Get component type */
    GpuComponentType type() const override
    {
        return GpuComponentType::Output;
    }

private:
    std::unique_ptr<ClTemplateStore> _component_writer;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTSTORE */
