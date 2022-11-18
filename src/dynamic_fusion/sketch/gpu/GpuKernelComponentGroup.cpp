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

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
bool GpuKernelComponentGroup::add_component(ComponentPtr component)
{
    // note: Constraint 1 is guaranteed as a precondition
    // Constraint 2
    if(component->type() != GpuComponentType::Output && _components.size() >= max_fused_components)
    {
        return false;
    }
    // Constraint 3.1: Pattern: (Unfusable + Output)
    if(!_components.empty() && get_root_component()->type() == GpuComponentType::Unfusable && component->type() != GpuComponentType::Output)
    {
        return false;
    }
    // Constraint 3.2
    if(!_components.empty() && (component->type() != GpuComponentType::Simple && component->type() != GpuComponentType::Output))
    {
        return false;
    }
    // Constraint 3.3: Disallow multiple output components
    if(!_components.empty() && get_last_component()->type() == GpuComponentType::Output && component->type() == GpuComponentType::Output)
    {
        return false;
    }
    // Constraint 4
    if(component->type() != GpuComponentType::Unfusable && component->tensors().get_const_dst_tensors().size() != 1U)
    {
        return false;
    }
    // Constraint 5
    if(!_components.empty() && !(get_root_component()->properties() == component->properties()))
    {
        return false;
    }
    // Constraint 7
    if(!_components.empty())
    {
        const auto root_dst_tensors = get_root_component()->tensors().get_const_dst_tensors();
        ARM_COMPUTE_ERROR_ON(root_dst_tensors.empty());
        const auto first_dst_tensor = root_dst_tensors[0];
        const auto dst_tensors      = component->tensors().get_const_dst_tensors();
        for(const auto &t : root_dst_tensors)
        {
            if(detail::have_different_dimensions(t->tensor_shape(), first_dst_tensor->tensor_shape(), 0))
            {
                return false;
            }
        }
        for(const auto &t : dst_tensors)
        {
            if(detail::have_different_dimensions(t->tensor_shape(), first_dst_tensor->tensor_shape(), 0))
            {
                return false;
            }
        }
    }
    // Constraint 8
    if(!_components.empty())
    {
        const auto root_dst_tensors = get_root_component()->tensors().get_const_dst_tensors();
        ARM_COMPUTE_ERROR_ON(root_dst_tensors.empty());
        const auto first_dst_tensor_layout = root_dst_tensors[0]->data_layout();
        const auto dst_tensors             = component->tensors().get_const_dst_tensors();
        for(const auto &t : root_dst_tensors)
        {
            if(t->data_layout() != first_dst_tensor_layout)
            {
                return false;
            }
        }
        for(const auto &t : dst_tensors)
        {
            if(t->data_layout() != first_dst_tensor_layout)
            {
                return false;
            }
        }
    }
    // Constraint 9
    if(component->tensors().get_const_dst_tensors().size() >= max_dst_tensors)
    {
        return false;
    }
    // Constraint 9 corollary
    if(component->type() == GpuComponentType::Output && _components.size() >= max_fused_components + max_dst_tensors)
    {
        return false;
    }
    _components.push_back(component);
    return true;
}

std::vector<const ITensorInfo *> GpuKernelComponentGroup::get_src_tensors() const
{
    if(_components.empty())
    {
        return {};
    }
    auto src_tensors     = _components[0]->tensors().get_const_src_tensors();
    auto prev_dst_tensor = _components[0]->tensors().get_const_dst_tensors()[0]; // PRE: Only one dst tensor per component
    for(unsigned int i = 1; i < _components.size(); ++i)
    {
        auto cur_src_tensors = _components[i]->tensors().get_const_src_tensors();
        for(const auto src_tensor : cur_src_tensors)
        {
            if(src_tensor->id() == prev_dst_tensor->id())
            {
                continue; // Skip "intermediate" tensors. I.e. tensors that are used to link between two components
            }
            src_tensors.push_back(src_tensor);
        }
        prev_dst_tensor = _components[i]->tensors().get_const_dst_tensors()[0]; // PRE: Only one dst tensor per component
    }

    return src_tensors;
}

std::vector<const ITensorInfo *> GpuKernelComponentGroup::get_dst_tensors() const
{
    if(_components.empty())
    {
        return {};
    }
    const auto                       dst_tensor_ptrs = _components[_components.size() - 1]->tensors().get_const_dst_tensors();
    std::vector<const ITensorInfo *> dst_tensors;
    for(auto tensor_ptr : dst_tensor_ptrs)
    {
        dst_tensors.push_back(tensor_ptr);
    }
    return dst_tensors;
}

std::vector<const ITensorInfo *> GpuKernelComponentGroup::get_argument_tensors() const
{
    std::vector<const ITensorInfo *> arguments;
    const auto                       src_tensors = get_src_tensors();
    const auto                       dst_tensors = get_dst_tensors();
    arguments.reserve(src_tensors.size() + dst_tensors.size());
    arguments.insert(arguments.end(), src_tensors.begin(), src_tensors.end());
    arguments.insert(arguments.end(), dst_tensors.begin(), dst_tensors.end());
    return arguments;
}

GpuKernelComponentGroup::ComponentPtr GpuKernelComponentGroup::get_root_component() const
{
    if(empty())
    {
        return nullptr;
    }
    return _components[0];
}

GpuKernelComponentGroup::ComponentPtr GpuKernelComponentGroup::get_last_component() const
{
    if(empty())
    {
        return nullptr;
    }
    return _components[_components.size() - 1];
}

GpuKernelComponentGroup::ComponentPtr GpuKernelComponentGroup::get_previous_component(ComponentId id) const
{
    if(empty())
    {
        return nullptr;
    }
    // Get the index of the requested component
    size_t ind = 0;
    for(const auto c : _components)
    {
        if(c->id() == id)
        {
            break;
        }
        ind++;
    }
    if(ind == 0 || ind >= _components.size())
    {
        return nullptr;
    }
    return _components[ind - 1];
}

bool GpuKernelComponentGroup::is_intermediate_tensor(const ITensorInfo *tensor) const
{
    return is_tensor_in(tensor, get_interm_tensors());
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

std::vector<const ITensorInfo *> GpuKernelComponentGroup::get_interm_tensors() const
{
    std::vector<const ITensorInfo *> interm_tensors{};
    for(unsigned int i = 0; i + 1 < _components.size(); ++i)
    {
        auto interm_tensor = _components[i]->tensors().get_const_dst_tensors()[0];
        interm_tensors.push_back(interm_tensor); // PRE: Only one dst tensor per component
    }

    return interm_tensors;
}

bool GpuKernelComponentGroup::is_tensor_in(const ITensorInfo *tensor, const std::vector<const ITensorInfo *> tensors)
{
    for(auto t : tensors)
    {
        if(tensor->id() == t->id())
        {
            return true;
        }
    }
    return false;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
