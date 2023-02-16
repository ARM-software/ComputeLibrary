/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTLOGITS1DMAXSHIFTEXPSUM
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTLOGITS1DMAXSHIFTEXPSUM

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
class ClTemplateLogits1DMaxShiftExpSum;

/** Component to calculate max-shifted exponentials and their sum
 *
 *  1D example:
 *      input:  [x1, x2, ... , xn], shape: (1 x d)
 *
 *      Let max(x1...xn) = m
 *
 *      (output) sum: [exp(x1-m) + ... + exp(xn-m)], shape: (1 x 1)
 *      (output) dst: [exp(x1-m) ... exp(xn-m)], shape: (1 x d)
 *
 *  This component is used by the softmax operator. The subsequent
 *  operation normalizes dst with sum, therefore the max-shifting
 *  since exp(m) will be cancelled in numerator and denominator.
*/
class ClComponentLogits1DMaxShiftExpSum final : public IGpuKernelComponent
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
     * - ACL_DST_0: Output
     * - ACL_DST_1: Output
     *
     * Tensor argument constness:
     * - ACL_SRC_0: Const
     * - ACL_DST_0: Const
     * - ACL_DST_1: Const
     *
     * Valid data layouts:
     * - All
     *
     ** Valid data type configurations:
     * |ACL_SRC_0  |ACL_DST_0  |ACL_DST_1  |
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
     * Similar to @ref ClComponentLogits1DMaxShiftExpSum::validate()
     */
    ClComponentLogits1DMaxShiftExpSum(ComponentId                      id,
                                      const Properties                &properties,
                                      const ArgumentPack<ITensorInfo> &tensors,
                                      const Attributes                &attributes);

    /** Destructor */
    ~ClComponentLogits1DMaxShiftExpSum() override;
    /** Prevent instances of this class from being copy constructed */
    ClComponentLogits1DMaxShiftExpSum(const ClComponentLogits1DMaxShiftExpSum &component) = delete;
    /** Prevent instances of this class from being copied */
    ClComponentLogits1DMaxShiftExpSum &operator=(const ClComponentLogits1DMaxShiftExpSum &component) = delete;
    /** Allow instances of this class to be move constructed */
    ClComponentLogits1DMaxShiftExpSum(ClComponentLogits1DMaxShiftExpSum &&component) = default;
    /** Allow instances of this class to be moved */
    ClComponentLogits1DMaxShiftExpSum &operator=(ClComponentLogits1DMaxShiftExpSum &&component) = default;
    /** Get template writer for the component */
    const IGpuTemplateComponentWriter *template_writer() const override;
    /** Get component type */
    GpuComponentType type() const override
    {
        return GpuComponentType::Unfusable;
    }

private:
    std::unique_ptr<ClTemplateLogits1DMaxShiftExpSum> _component_writer;
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_CL_CLCOMPONENTLOGITS1DMAXSHIFTEXPSUM */
