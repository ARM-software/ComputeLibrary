/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_BACKEND_REGISTRAR_H
#define ARM_COMPUTE_GRAPH_BACKEND_REGISTRAR_H

#include "arm_compute/graph/Types.h"
#include "arm_compute/graph/backends/BackendRegistry.h"

#include <utility>

namespace arm_compute
{
namespace graph
{
namespace backends
{
namespace detail
{
/** Helper class to statically register a backend */
template <typename T>
class BackendRegistrar final
{
public:
    /** Add a new backend to the backend registry
     *
     * @param[in] target Execution target
     */
    BackendRegistrar(Target target);
};

template <typename T>
inline BackendRegistrar<T>::BackendRegistrar(Target target)
{
    BackendRegistry::get().add_backend<T>(target);
}
} // namespace detail
} // namespace backends
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_BACKEND_REGISTRAR_H */