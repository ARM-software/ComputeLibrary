/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/graph2/GraphContext.h"
#include <arm_compute/graph2.h>

namespace arm_compute
{
namespace graph2
{
GraphContext::GraphContext()
    : _tunable(false), _memory_managed(false), _memory_managers()
{
}

void GraphContext::enable_tuning(bool enable_tuning)
{
    _tunable = enable_tuning;
}

bool GraphContext::is_tuning_enabled() const
{
    return _tunable;
}

void GraphContext::enable_memory_managenent(bool enable_mm)
{
    _memory_managed = enable_mm;
}

bool GraphContext::is_memory_management_enabled()
{
    return _memory_managed;
}

bool GraphContext::insert_memory_management_ctx(MemoryManagerContext &&memory_ctx)
{
    Target target = memory_ctx.target;
    if(target == Target::UNSPECIFIED || _memory_managers.find(target) != std::end(_memory_managers))
    {
        return false;
    }

    _memory_managers[target] = std::move(memory_ctx);
    return true;
}

MemoryManagerContext *GraphContext::memory_management_ctx(Target target)
{
    return (_memory_managers.find(target) != std::end(_memory_managers)) ? &_memory_managers[target] : nullptr;
}

void GraphContext::finalize()
{
    for(auto &mm_obj : _memory_managers)
    {
        if(mm_obj.second.mm != nullptr)
        {
            mm_obj.second.mm->finalize();
        }
    }
}
} // namespace graph2
} // namespace arm_compute