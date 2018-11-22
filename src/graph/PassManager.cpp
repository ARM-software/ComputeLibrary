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
#include "arm_compute/graph/PassManager.h"

#include "arm_compute/graph/Logger.h"

namespace arm_compute
{
namespace graph
{
PassManager::PassManager()
    : _passes()
{
}

const std::vector<std::unique_ptr<IGraphMutator>> &PassManager::passes() const
{
    return _passes;
}

IGraphMutator *PassManager::pass(size_t index)
{
    return (index >= _passes.size()) ? nullptr : _passes.at(index).get();
}

void PassManager::append(std::unique_ptr<IGraphMutator> pass, bool conditional)
{
    if(pass && conditional)
    {
        ARM_COMPUTE_LOG_GRAPH_VERBOSE("Appending mutating pass : " << pass->name() << std::endl);
        _passes.push_back(std::move(pass));
    }
}

void PassManager::clear()
{
    _passes.clear();
}

void PassManager::run_all(Graph &g)
{
    for(auto &pass : _passes)
    {
        if(pass)
        {
            ARM_COMPUTE_LOG_GRAPH_INFO("Running mutating pass : " << pass->name() << std::endl);
            pass->mutate(g);
        }
    }
}

void PassManager::run(Graph &g, size_t index)
{
    if(index >= _passes.size())
    {
        return;
    }

    auto &pass = _passes.at(index);

    if(pass != nullptr)
    {
        pass->mutate(g);
    }
}
} // namespace graph
} // namespace arm_compute