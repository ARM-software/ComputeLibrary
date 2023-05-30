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

#ifndef CKW_INCLUDE_ACL_ACLSCOPEDKERNELWRITER_H
#define CKW_INCLUDE_ACL_ACLSCOPEDKERNELWRITER_H

#include <cstdint>

class AclKernelWriter;

/** Helper to automatically manage kernel writer ID space. */
class AclScopedKernelWriter
{
public:
    /** Initialize a new instance of @ref AclScopedKernelWriter class. */
    explicit AclScopedKernelWriter(AclKernelWriter *writer);

    /** Create a new scope from the specified scoped kernel writer. */
    AclScopedKernelWriter(const AclScopedKernelWriter &other);

    /** Assignment is disallowed. */
    AclScopedKernelWriter &operator=(const AclScopedKernelWriter &) = delete;

    /** Access the underlying kernel writer. */
    AclKernelWriter *operator->();

    /** Access the underlying kernel writer. */
    const AclKernelWriter *operator->() const;

    /** Get the kernel writer. */
    AclKernelWriter *writer();

    /** Get the kernel writer. */
    const AclKernelWriter *writer() const;

private:
    AclKernelWriter *_writer;
    int32_t          _parent_id_space;
};

#endif // CKW_INCLUDE_ACL_ACLSCOPEDKERNELWRITER_H
