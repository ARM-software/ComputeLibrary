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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTGROUP
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTGROUP

#include "components/Types.h"

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <set>
#include <map>

namespace arm_compute
{
/** Forward declaration */
class ITensorInfo;
namespace experimental
{
namespace dynamic_fusion
{
class IGpuKernelComponent;
/** A group of gpu kernel components to be fused together
 * PRECONDITIONS:
 * 1. Fusion is limited to a linear sequence of kernel components
 * INVARIANTS:
 * @note These preconditions and invariants are exactly the same as fusion constraints for kernel components
 * 2. Max number of components that can be fused is @ref GpuKernelComponentGroup::max_fused_components (
 *        excluding any output or input (if any) components.
 *        The max number of output components are bound by the maximum number of dst tensors allowed for a component / component group
 *    )
 * 3. The fusion is subject to the pattern: (Complex + Simple * | Simple + Simple * | Un-fusable) + Output?
 * 4. All components but unfusable, have exactly 1 dst tensor
 * 5. All fused components share the same @ref IGpuKernelComponent::Properties ( @ref UnitWorkloadStage etc. )
 * 6. All fused components share the same tunable parameters like tile size
 * 7. All fused components share the same dst tensor shape
 * 8. All fused components' tensors share the same @ref DataLayout
 * 9. Maximum number of dst tensors allowed for an component (including unfusable) / component group is @ref GpuKernelComponentGroup::max_dst_tensors
 *      This has an impact on the total number of components supported, which = max_fused_components + max_dst_tensors
 */
class GpuKernelComponentGroup
{
public:
    using ComponentPtr = IGpuKernelComponent *;
    /** Maximum number of components that can be fused into the same component group
     */
    static constexpr size_t max_fused_components = 64;
    /** Maximum number of dst tensors allowed for a component / component
     */
    static constexpr size_t max_dst_tensors = 8;

public:
    /** Default constructor */
    GpuKernelComponentGroup() = default;
    /** Allow instances of this class to be copy constructed */
    GpuKernelComponentGroup(const GpuKernelComponentGroup &) = default;
    /** Allow instances of this class to be copied */
    GpuKernelComponentGroup &operator=(const GpuKernelComponentGroup &) = default;
    /** Allow instances of this class to be move constructed */
    GpuKernelComponentGroup(GpuKernelComponentGroup &&) = default;
    /** Allow instances of this class to be moved */
    GpuKernelComponentGroup &operator=(GpuKernelComponentGroup &&) = default;
    /** Add a component pointer into the group
     * If the operation fails, then no change is made to the group
     *
     * @param[in] component Pointer to the component to be added
     *
     * @return true      If the operation is successful
     * @return false     If the operation fails
     */
    bool add_component(ComponentPtr component);
    /** Optimize and pre-compute information about the component group */
    void finalize();
    /** Get one of the destination tensors of this group */
    const ITensorInfo *get_any_dst_tensor() const;
    /** Get tensor argument of this group
     *  A tensor is an argument if it is a source or destination tensor to the group
     */
    std::vector<const ITensorInfo *> get_argument_tensors() const;
    /** Get the root (first) component of this group */
    ComponentPtr get_root_component() const;
    /** Check if a @ref ITensorInfo is an "intermediate" tensor of the group
     *
     * An intermediate tensor is any tensor that is not an argument.
     *
     * @param[in] tensor @ref ITensorInfo to be looked up
     *
     * @return true  If @p tensor is an intermediate tensor
     * @return false  Otherwise
     */
    bool is_intermediate_tensor(const ITensorInfo *tensor) const;
    /** Check if an @ref ITensorInfo is an input tensor of the group.
     *
     * @param[in] tensor @ref ITensorInfo to be looked up.
     *
     * @return true if @p tensor is an input tensor of the group, otherwise false.
     */
    bool is_input_tensor(const ITensorInfo *tensor) const;
    /** Get the list of temporary tiles that need to be declared */
    std::vector<const ITensorInfo *> get_tiles() const;
    /** Get the shared tile that can be used to store temporary data of the specified tensor.
     *
     * @param[in] tensor @ref ITensorInfo to be looked up.
     *
     * @return @ref ITensorInfo that is used to store temporary data of @p tensor.
     **/
    const ITensorInfo *get_tile_for_tensor(const ITensorInfo *tensor) const;
    /** Get the number of components within the group */
    size_t size() const;
    /** Check if the component group is empty */
    bool empty() const;
    ComponentPtr &operator[](size_t index);
    const ComponentPtr &operator[](size_t index) const;
    typename std::vector<ComponentPtr>::iterator       begin();
    typename std::vector<ComponentPtr>::iterator       end();
    typename std::vector<ComponentPtr>::const_iterator begin() const;
    typename std::vector<ComponentPtr>::const_iterator end() const;
    typename std::vector<ComponentPtr>::const_iterator cbegin() const;
    typename std::vector<ComponentPtr>::const_iterator cend() const;

private:
    std::vector<ComponentPtr> _components{};

    bool _finalized{ false };

    std::vector<const ITensorInfo *> _argument_tensors{};
    std::set<const ITensorInfo *> _input_tensors{};
    std::set<const ITensorInfo *> _interm_tensors{};
    const ITensorInfo *_any_output_tensor{ nullptr };
    std::vector<const ITensorInfo *> _tiles{};
    std::map<const ITensorInfo *, const ITensorInfo *> _tile_map{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELCOMPONENTGROUP */
