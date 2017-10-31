/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_ERROR_H__
#define __ARM_COMPUTE_GRAPH_ERROR_H__

#include "arm_compute/graph/ITensorObject.h"

namespace arm_compute
{
namespace graph
{
/** Evaluate if a tensor object is null. If the condition is true then an error message is printed and an exception thrown
 *
 * @param[in] function       Function in which the error occurred.
 * @param[in] file           Name of the file where the error occurred.
 * @param[in] line           Line on which the error occurred.
 * @param[in] tensor_object  Tensor object to evaluate
 * @param[in] tensor_objects (Optional) Further allowed tensor objects.
 */
template <typename... Ts>
void error_on_unallocated_tensor_object(const char *function, const char *file, int line,
                                        const ITensorObject *tensor_object, Ts... tensor_objects)
{
    ARM_COMPUTE_UNUSED(function);
    ARM_COMPUTE_UNUSED(file);
    ARM_COMPUTE_UNUSED(line);
    ARM_COMPUTE_UNUSED(tensor_object);

    ARM_COMPUTE_ERROR_ON_LOC(tensor_object == nullptr || tensor_object->tensor() == nullptr, function, file, line);

    const std::array<const ITensorObject *, sizeof...(Ts)> tensor_objects_array{ { std::forward<Ts>(tensor_objects)... } };
    ARM_COMPUTE_UNUSED(tensor_objects_array);

    ARM_COMPUTE_ERROR_ON_LOC(std::any_of(tensor_objects_array.begin(), tensor_objects_array.end(), [&](const ITensorObject * tensor_obj)
    {
        return (tensor_obj == nullptr || tensor_object->tensor() == nullptr);
    }),
    function, file, line);
}
#define ARM_COMPUTE_ERROR_ON_UNALLOCATED_TENSOR_OBJECT(...) ::arm_compute::graph::error_on_unallocated_tensor_object(__func__, __FILE__, __LINE__, __VA_ARGS__)
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_ERROR_H__ */
