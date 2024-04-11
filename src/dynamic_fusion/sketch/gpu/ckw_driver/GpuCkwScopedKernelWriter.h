/*
 * Copyright (c) 2023 Arm Limited.
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

#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWSCOPEDKERNELWRITER_H
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWSCOPEDKERNELWRITER_H

#include <cstdint>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{

class GpuCkwKernelWriter;

/** Helper to automatically manage kernel writer ID space. */
class GpuCkwScopedKernelWriter
{
public:
    /** Initialize a new instance of @ref GpuCkwScopedKernelWriter class. */
    explicit GpuCkwScopedKernelWriter(GpuCkwKernelWriter *writer);

    /** Create a new scope from the specified scoped kernel writer. */
    GpuCkwScopedKernelWriter(const GpuCkwScopedKernelWriter &other);

    /** Assignment is disallowed. */
    GpuCkwScopedKernelWriter &operator=(const GpuCkwScopedKernelWriter &) = delete;

    /** Access the underlying kernel writer. */
    GpuCkwKernelWriter *operator->();

    /** Access the underlying kernel writer. */
    const GpuCkwKernelWriter *operator->() const;

    /** Get the kernel writer. */
    GpuCkwKernelWriter *writer();

    /** Get the kernel writer. */
    const GpuCkwKernelWriter *writer() const;

private:
    GpuCkwKernelWriter *_writer;
    int32_t             _parent_id_space;
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif // ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWSCOPEDKERNELWRITER_H
