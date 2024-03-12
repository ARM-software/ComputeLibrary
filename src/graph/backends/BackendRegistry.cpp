/*
 * Copyright (c) 2018 Arm Limited.
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
#include "arm_compute/graph/backends/BackendRegistry.h"

using namespace arm_compute::graph::backends;

namespace arm_compute
{
namespace graph
{
namespace backends
{
BackendRegistry::BackendRegistry() : _registered_backends()
{
}

BackendRegistry &BackendRegistry::get()
{
    static BackendRegistry instance;
    return instance;
}

IDeviceBackend *BackendRegistry::find_backend(Target target)
{
    ARM_COMPUTE_ERROR_ON(!contains(target));
    return _registered_backends[target].get();
}

IDeviceBackend &BackendRegistry::get_backend(Target target)
{
    IDeviceBackend *backend = find_backend(target);
    ARM_COMPUTE_ERROR_ON_MSG(!backend, "Requested backend doesn't exist!");
    ARM_COMPUTE_ERROR_ON_MSG(!backend->is_backend_supported(), "Requested backend isn't supported");
    return *backend;
}

bool BackendRegistry::contains(Target target) const
{
    auto it = _registered_backends.find(target);
    return (it != _registered_backends.end());
}

const std::map<Target, std::unique_ptr<IDeviceBackend>> &BackendRegistry::backends() const
{
    return _registered_backends;
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
