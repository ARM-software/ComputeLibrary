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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTLOGITS1DNORM
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTLOGITS1DNORM

#include "arm_compute/dynamic_fusion/sketch/attributes/SoftmaxAttributes.h"
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
class ClTemplateLogits1DNorm;

/** Component to calculate the final step of the Softmax Layer
 * where each logit value is multiplied by the inverse of the sum of the logits.
 *
 *  1D example:
 *
 *      (input)  src: [x1 x2 ... xn], shape: (1 x d)
 *      (input)  sum: [x1 + x2 + ... + xn], shape: (1 x 1)
 *      (output) dst: [x1/sum x2/sum ... xn/sum], shape: (1 x d)
 *
 *  This component is used by the softmax operator to get the final result.
*/
class ClComponentLogits1DNorm final : public IGpuKernelComponent
{
public:
    /** Attributes are a set of backend-agnostic parameters that define what a component does */
    using Attributes = SoftmaxAttributes;

    /** Validate the component
     *
     * @param[in] properties Component properties @ref Properties
     * @param[in] tensors    Tensor arguments to the component
     * @param[in] attributes Component attributes @ref Attributes
     *
     * @return Status        Validation results
     *
     * Tensor argument names:
     * - ACL_SRC_0: Input
     * - ACL_SRC_1: Input
     * - ACL_DST_0: Output
     *
     * Tensor argument constness:
     * - ACL_SRC_0: Const
     * - ACL_SRC_1: Const
     * - ACL_DST_0: Const
     *
     * Valid data layouts:
     * - All
     *
     ** Valid data type configurations:
     * |ACL_SRC_0  |ACL_SRC_1  |ACL_DST_0  |
     * |:----------|:----------|:----------|
     * |F16        | F16       | F16       |
     * |F32        | F32       | F32       |
     */
    static Status validate(
        const Properties                &properties,
        const ArgumentPack<ITensorInfo> &tensors,
        const Attributes                &attributes);

    /** Constructor
     *
     * Similar to @ref ClComponentLogits1DNorm::validate()
     */
    ClComponentLogits1DNorm(ComponentId                      id,
                            const Properties                &properties,
                            const ArgumentPack<ITensorInfo> &tensors,
                            const Attributes                &attributes);

    /** Destructor */
    ~ClComponentLogits1DNorm() override;
    /** Prevent instances of this class from being copy constructed */
    ClComponentLogits1DNorm(const ClComponentLogits1DNorm &component) = delete;
    /** Prevent instances of this class from being copied */
    ClComponentLogits1DNorm &operator=(const ClComponentLogits1DNorm &component) = delete;
    /** Allow instances of this class to be move constructed */
    ClComponentLogits1DNorm(ClComponentLogits1DNorm &&component) = default;
    /** Allow instances of this class to be moved */
    ClComponentLogits1DNorm &operator=(ClComponentLogits1DNorm &&component) = default;
    /** Get template writer for the component */
    const IGpuTemplateComponentWriter *template_writer() const override;
    /** Get component type */
    GpuComponentType type() const override
    {
        return GpuComponentType::Unfusable;
    }

private:
    std::unique_ptr<ClTemplateLogits1DNorm> _component_writer;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTLOGITS1DNORM */
