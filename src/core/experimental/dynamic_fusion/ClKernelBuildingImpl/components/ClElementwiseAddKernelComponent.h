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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION

#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMPONENTS_CLELEMENTWISEADDKERNELCOMPONENT_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMPONENTS_CLELEMENTWISEADDKERNELCOMPONENT_H

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/Common.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class ClElementwiseAddKernelComponent : public IClKernelComponent
{
public:
    ClElementwiseAddKernelComponent(ClKernelBlueprint *blueprint, const Link &lhs, const Link &rhs, const Link &dst)
        : IClKernelComponent(blueprint), _lhs{ lhs }, _rhs{ rhs }, _dst{ dst }
    {
    }

    ComponentType         get_component_type() const override;
    std::set<std::string> get_headers_list() const override;
    std::string           get_component_code() const override;
    Window                get_window() const override;
    CLBuildOptions        generate_build_options() const override;
    std::string           generate_config_id() const override;

    virtual std::vector<Link> get_links() const override
    {
        return { _lhs, _rhs, _dst };
    }

    virtual TagLUT get_tag_lut(const SharedVarTable &vtable) const override;
    virtual void allocate_shared_vars(SharedVarTable &vtable) const override;

    virtual std::string name() const override
    {
        return "eltwise_add_" + std::to_string(id());
    }

private:
    Link _lhs{};
    Link _rhs{};
    Link _dst{};
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif // ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMPONENTS_CLELEMENTWISEADDKERNELCOMPONENT_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */