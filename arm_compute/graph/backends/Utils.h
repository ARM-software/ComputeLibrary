/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_BACKENDS_UTILS_H
#define ARM_COMPUTE_GRAPH_BACKENDS_UTILS_H

#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/IWeightsManager.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Creates and configures a named function
 *
 * @param[in] name Name of the function
 * @param[in] args Function arguments
 *
 * @return  A configured backend function
 */
template <typename FunctionType, typename FunctionNameType, typename... ParameterType>
std::tuple<std::unique_ptr<arm_compute::IFunction>, FunctionNameType> create_named_function(FunctionNameType name, ParameterType... args)
{
    auto f = std::make_unique<FunctionType>();
    f->configure(std::forward<ParameterType>(args)...);
    return std::make_pair(std::move(f), name);
}

/** Creates and configures a named function
 *
 * @param[in] name Name of the function
 * @param[in] mm   Memory manager to use
 * @param[in] args Function arguments
 *
 * @return  A configured backend function
 */
template <typename FunctionType, typename FunctionNameType, typename MemoryManagerType, typename... ParameterType>
std::tuple<std::unique_ptr<arm_compute::IFunction>, FunctionNameType> create_named_memory_managed_function(FunctionNameType name,
                                                                                                           MemoryManagerType mm,
                                                                                                           ParameterType... args)
{
    auto f = std::make_unique<FunctionType>(mm);
    f->configure(std::forward<ParameterType>(args)...);
    return std::make_pair(std::move(f), name);
}

/** Checks if an operation is in place
 *
 * @param[in] input  Pointer to input
 * @param[in] output Pointer to output
 *
 * @return True if output is nullptr or input is equal to the output, else false
 */
inline bool is_in_place_operation(void *input, void *output)
{
    return (output == nullptr) || (input == output);
}

/** Returns the memory manager for a given target
 *
 * @param[in] ctx    Graph context containing memory management metadata
 * @param[in] target Target to retrieve the memory manager from
 *
 * @return The memory manager for the given target else false
 */
inline std::shared_ptr<IMemoryManager> get_memory_manager(GraphContext &ctx, Target target)
{
    bool enabled = ctx.config().use_function_memory_manager && (ctx.memory_management_ctx(target) != nullptr);
    return enabled ? ctx.memory_management_ctx(target)->intra_mm : nullptr;
}

/** Returns the weights manager for a given target
 *
 * @param[in] ctx    Graph context containing weight management metadata
 * @param[in] target Target to retrieve the weights manager from
 *
 * @return The weights manager for the given target else false
 */
inline std::shared_ptr<IWeightsManager> get_weights_manager(GraphContext &ctx, Target target)
{
    bool enabled = ctx.config().use_function_weights_manager && (ctx.weights_management_ctx(target) != nullptr);
    return enabled ? ctx.weights_management_ctx(target)->wm : nullptr;
}
} // namespace backends
} // namespace graph
} // namespace arm_compute

#endif /* ARM_COMPUTE_GRAPH_BACKENDS_UTILS_H */
