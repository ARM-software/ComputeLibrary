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
#ifndef ARM_COMPUTE_GRAPH_BACKEND_REGISTRY_H
#define ARM_COMPUTE_GRAPH_BACKEND_REGISTRY_H

#include "arm_compute/graph/IDeviceBackend.h"
#include "arm_compute/graph/Types.h"

#include <map>
#include <memory>

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Registry holding all the supported backends */
class BackendRegistry final
{
public:
    /** Gets backend registry instance
     *
     * @return Backend registry instance
     */
    static BackendRegistry &get();
    /** Finds a backend in the registry
     *
     * @param[in] target Backend target
     *
     * @return Pointer to the backend interface if found, else nullptr
     */
    IDeviceBackend *find_backend(Target target);

    /** Get a backend from the registry
     *
     * The backend must be present and supported.
     *
     * @param[in] target Backend target
     *
     * @return Reference to the backend interface
     */
    IDeviceBackend &get_backend(Target target);
    /** Checks if a backend for a given target exists
     *
     * @param[in] target Execution target
     *
     * @return True if exists else false
     */
    bool contains(Target target) const;
    /** Backends accessor
     *
     * @return Map containing the registered backends
     */
    const std::map<Target, std::unique_ptr<IDeviceBackend>> &backends() const;
    /** Registers a backend to the registry
     *
     * @param[in] target Execution target to register for
     */
    template <typename T>
    void add_backend(Target target);

private:
    /** Default Constructor */
    BackendRegistry();

private:
    std::map<Target, std::unique_ptr<IDeviceBackend>> _registered_backends;
};

template <typename T>
inline void BackendRegistry::add_backend(Target target)
{
    _registered_backends[target] = std::make_unique<T>();
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_BACKEND_REGISTRY_H */
