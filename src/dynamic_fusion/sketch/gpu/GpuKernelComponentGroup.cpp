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
#include "GpuKernelComponentGroup.h"

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Validate.h"

#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"

#include <algorithm>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
bool GpuKernelComponentGroup::add_component(ComponentPtr component)
{
    ARM_COMPUTE_ERROR_ON_MSG(_finalized, "The component group has been finalized and cannot be altered.");

    // note: Constraint 1 is guaranteed as a precondition
    // Constraint 2
    if (component->type() != GpuComponentType::Output && _components.size() >= max_fused_components)
    {
        return false;
    }
    // Constraint 3.1: Pattern: (Unfusable + Output)
    if (!_components.empty() && get_root_component()->type() == GpuComponentType::Unfusable &&
        component->type() != GpuComponentType::Output)
    {
        return false;
    }
    // Constraint 3.2
    if (!_components.empty() &&
        (component->type() != GpuComponentType::Simple && component->type() != GpuComponentType::Output))
    {
        return false;
    }
    // Constraint 4
    if (component->type() != GpuComponentType::Unfusable && component->tensors().get_const_dst_tensors().size() != 1U)
    {
        return false;
    }
    // Constraint 5
    if (!_components.empty() && !(get_root_component()->properties() == component->properties()))
    {
        return false;
    }
    // Constraint 7
    if (!_components.empty())
    {
        const auto root_dst_tensors = get_root_component()->tensors().get_const_dst_tensors();
        ARM_COMPUTE_ERROR_ON(root_dst_tensors.empty());
        const auto first_dst_tensor = root_dst_tensors[0];
        const auto dst_tensors      = component->tensors().get_const_dst_tensors();
        for (const auto &t : root_dst_tensors)
        {
            if (detail::have_different_dimensions(t->tensor_shape(), first_dst_tensor->tensor_shape(), 0))
            {
                return false;
            }
        }
        for (const auto &t : dst_tensors)
        {
            if (detail::have_different_dimensions(t->tensor_shape(), first_dst_tensor->tensor_shape(), 0))
            {
                return false;
            }
        }
    }
    // Constraint 8
    if (!_components.empty())
    {
        const auto root_dst_tensors = get_root_component()->tensors().get_const_dst_tensors();
        ARM_COMPUTE_ERROR_ON(root_dst_tensors.empty());
        const auto first_dst_tensor_layout = root_dst_tensors[0]->data_layout();
        const auto dst_tensors             = component->tensors().get_const_dst_tensors();
        for (const auto &t : root_dst_tensors)
        {
            if (t->data_layout() != first_dst_tensor_layout)
            {
                return false;
            }
        }
        for (const auto &t : dst_tensors)
        {
            if (t->data_layout() != first_dst_tensor_layout)
            {
                return false;
            }
        }
    }
    // Constraint 9
    if (component->tensors().get_const_dst_tensors().size() >= max_dst_tensors)
    {
        return false;
    }
    // Constraint 9 corollary
    if (component->type() == GpuComponentType::Output && _components.size() >= max_fused_components + max_dst_tensors)
    {
        return false;
    }
    _components.push_back(component);
    return true;
}

void GpuKernelComponentGroup::finalize()
{
    if (_finalized)
    {
        return;
    }

    _finalized = true;

    std::set<const ITensorInfo *>                                   output_tensors;
    std::map<const ITensorInfo *, std::vector<const ITensorInfo *>> possible_tile_map;
    std::map<const ITensorInfo *, int32_t>                          tile_usages;

    for (auto component : _components)
    {
        const auto tensors     = component->tensors();
        const auto src_tensors = tensors.get_const_src_tensors();
        const auto dst_tensors = tensors.get_const_dst_tensors();

        // Detect input, output and intermediate tensors.
        for (auto tensor : src_tensors)
        {
            const auto output_tensors_it = output_tensors.find(tensor);

            if (output_tensors_it != output_tensors.end())
            {
                // This tensor is the output of another operator.
                // It must be marked as intermediate tensor.
                output_tensors.erase(output_tensors_it);
                _interm_tensors.insert(tensor);
            }
            else if (_interm_tensors.find(tensor) == _interm_tensors.end())
            {
                _input_tensors.insert(tensor);

                tile_usages[tensor] = 0;
                possible_tile_map.emplace(tensor, std::vector<const ITensorInfo *>());
            }
        }

        for (auto tensor : dst_tensors)
        {
            ARM_COMPUTE_ERROR_ON(_input_tensors.find(tensor) != _input_tensors.end());
            ARM_COMPUTE_ERROR_ON(output_tensors.find(tensor) != output_tensors.end());
            ARM_COMPUTE_ERROR_ON(_interm_tensors.find(tensor) != _interm_tensors.end());
            output_tensors.insert(tensor);

            tile_usages[tensor] = 0;
            possible_tile_map.emplace(tensor, std::vector<const ITensorInfo *>());
        }

        // Check if the output can overwrite the input tile.
        const auto component_type = component->type();
        if (component_type == GpuComponentType::Simple || component_type == GpuComponentType::Output)
        {
            ARM_COMPUTE_ERROR_ON(dst_tensors.size() != 1);

            const auto  dst_tensor = dst_tensors[0];
            const auto &dst_shape  = dst_tensor->tensor_shape();
            const auto &dst_type   = dst_tensor->data_type();

            tile_usages[dst_tensor] = 0;

            for (auto src_tensor : src_tensors)
            {
                const auto &src_shape = src_tensor->tensor_shape();
                const auto &src_type  = src_tensor->data_type();

                if (src_shape == dst_shape && src_type == dst_type)
                {
                    const auto tile_usages_it = tile_usages.find(src_tensor);
                    ARM_COMPUTE_ERROR_ON(tile_usages_it == tile_usages.end());

                    if (component_type == GpuComponentType::Simple || tile_usages_it->second > 0)
                    {
                        // Increase the number of tile usages unless this component is an output
                        // and the tile has not been shared with any component.
                        // (Reason: output component doesn't change the content of the tile)
                        ++tile_usages_it->second;
                    }

                    possible_tile_map[dst_tensor].push_back(src_tensor);
                }
            }
        }
        else
        {
            // Outputs of complex and unfusable components need dedicated tile.
            for (auto tensor : dst_tensors)
            {
                tile_usages[tensor] = 0;
            }
        }
    }

    // Find the smallest list of tiles that the intermediate tensors need to write to.
    for (auto tensor : _input_tensors)
    {
        _tile_map[tensor] = tensor;
    }

    for (auto component : _components)
    {
        const auto dst_tensors = component->tensors().get_const_dst_tensors();

        for (auto tensor : dst_tensors)
        {
            const auto target_tiles = possible_tile_map.at(tensor);
            _tile_map[tensor]       = tensor;

            for (auto target : target_tiles)
            {
                const auto num_usage = tile_usages[target];

                if (num_usage <= 1)
                {
                    // The target tile is consumed by only this operator, so we can reuse it
                    // for the destination tensor data.
                    _tile_map[tensor] = _tile_map.at(target);
                    break;
                }
            }
        }
    }

    for (auto tensor : output_tensors)
    {
        _tile_map[tensor] = tensor;
    }

    // All intermediate tensors that cannot be shared with any previous tensor
    // will need to be declared as tile variable.
    for (auto tensor_tile : _tile_map)
    {
        if (tensor_tile.first == tensor_tile.second && _interm_tensors.find(tensor_tile.first) != _interm_tensors.end())
        {
            _tiles.push_back(tensor_tile.first);
        }
    }

    std::set_union(_input_tensors.begin(), _input_tensors.end(), output_tensors.begin(), output_tensors.end(),
                   std::back_inserter(_argument_tensors));
    _any_output_tensor = *output_tensors.begin();
}

std::vector<const ITensorInfo *> GpuKernelComponentGroup::get_tiles() const
{
    ARM_COMPUTE_ERROR_ON_MSG(!_finalized, "The component group must have been finalized.");
    return _tiles;
}

const ITensorInfo *GpuKernelComponentGroup::get_tile_for_tensor(const ITensorInfo *tensor) const
{
    ARM_COMPUTE_ERROR_ON_MSG(!_finalized, "The component group must have been finalized.");

    if (_tile_map.find(tensor) != _tile_map.end())
    {
        return _tile_map.at(tensor);
    }

    return tensor;
}

const ITensorInfo *GpuKernelComponentGroup::get_any_dst_tensor() const
{
    ARM_COMPUTE_ERROR_ON_MSG(!_finalized, "The component group must have been finalized.");
    return _any_output_tensor;
}

std::vector<const ITensorInfo *> GpuKernelComponentGroup::get_argument_tensors() const
{
    ARM_COMPUTE_ERROR_ON_MSG(!_finalized, "The component group must have been finalized.");
    return _argument_tensors;
}

GpuKernelComponentGroup::ComponentPtr GpuKernelComponentGroup::get_root_component() const
{
    if (empty())
    {
        return nullptr;
    }
    return _components[0];
}

bool GpuKernelComponentGroup::is_intermediate_tensor(const ITensorInfo *tensor) const
{
    ARM_COMPUTE_ERROR_ON_MSG(!_finalized, "The component group must have been finalized.");
    return _interm_tensors.find(tensor) != _interm_tensors.end();
}

bool GpuKernelComponentGroup::is_input_tensor(const ITensorInfo *tensor) const
{
    ARM_COMPUTE_ERROR_ON_MSG(!_finalized, "The component group must have been finalized.");
    return _input_tensors.find(tensor) != _input_tensors.end();
}

size_t GpuKernelComponentGroup::size() const
{
    return _components.size();
}
bool GpuKernelComponentGroup::empty() const
{
    return _components.empty();
}
GpuKernelComponentGroup::ComponentPtr &GpuKernelComponentGroup::operator[](size_t index)
{
    return _components[index];
}
const GpuKernelComponentGroup::ComponentPtr &GpuKernelComponentGroup::operator[](size_t index) const
{
    return _components[index];
}
typename std::vector<GpuKernelComponentGroup::ComponentPtr>::iterator GpuKernelComponentGroup::begin()
{
    return _components.begin();
}
typename std::vector<GpuKernelComponentGroup::ComponentPtr>::iterator GpuKernelComponentGroup::end()
{
    return _components.end();
}
typename std::vector<GpuKernelComponentGroup::ComponentPtr>::const_iterator GpuKernelComponentGroup::begin() const
{
    return _components.cbegin();
}
typename std::vector<GpuKernelComponentGroup::ComponentPtr>::const_iterator GpuKernelComponentGroup::end() const
{
    return _components.cend();
}
typename std::vector<GpuKernelComponentGroup::ComponentPtr>::const_iterator GpuKernelComponentGroup::cbegin() const
{
    return _components.cbegin();
}
typename std::vector<GpuKernelComponentGroup::ComponentPtr>::const_iterator GpuKernelComponentGroup::cend() const
{
    return _components.cend();
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
