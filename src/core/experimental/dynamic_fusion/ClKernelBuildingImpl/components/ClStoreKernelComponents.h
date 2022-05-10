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

#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMPONENTS_CLSTOREKERNELCOMPONENTS_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMPONENTS_CLSTOREKERNELCOMPONENTS_H

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingImpl/Common.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
class ClStoreBlockBoundaryAwareKernelComponent : public IClKernelComponent
{
public:
    ClStoreBlockBoundaryAwareKernelComponent(ClKernelBlueprint *blueprint, const Link &src, const Link &dst)
        : IClKernelComponent(blueprint), _src{ src }, _dst{ dst }
    {
    }
    ComponentType  get_component_type() const override;
    std::string    get_component_code() const override;
    CLBuildOptions generate_build_options() const override;
    TagLUT get_tag_lut(const SharedVarTable &vtable) const override;
    void allocate_shared_vars(SharedVarTable &vtable) const override;

    virtual std::vector<Link> get_links() const override
    {
        return { _src, _dst };
    }

    virtual std::string name() const override
    {
        return "";
    }

private:
    Link _src{};
    Link _dst{};
};

class ClStoreIndirectWidthSelectKernelComponent : public IClKernelComponent
{
public:
    ClStoreIndirectWidthSelectKernelComponent(ClKernelBlueprint *blueprint, const Link &src, const Link &dst)
        : IClKernelComponent(blueprint), _src{ src }, _dst{ dst }
    {
    }
    ComponentType  get_component_type() const override;
    std::string    get_component_code() const override;
    CLBuildOptions generate_build_options() const override;
    virtual TagLUT get_tag_lut(const SharedVarTable &vtable) const override;
    void allocate_shared_vars(SharedVarTable &vtable) const override;

    virtual std::vector<Link> get_links() const override
    {
        return { _src, _dst };
    }

    virtual std::string name() const override
    {
        return "";
    }

private:
    Link _src{};
    Link _dst{};
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif // ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_COMPONENTS_CLSTOREKERNELCOMPONENTS_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */