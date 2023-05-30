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

#include "acl/AclScopedKernelWriter.h"
#include "acl/AclKernelWriter.h"

AclScopedKernelWriter::AclScopedKernelWriter(AclKernelWriter *writer)
    : _writer(writer), _parent_id_space(writer->id_space())
{
    _writer->next_id_space();
}

AclScopedKernelWriter::AclScopedKernelWriter(const AclScopedKernelWriter &other)
    : _writer(other._writer), _parent_id_space(other._writer->id_space())
{
    _writer->next_id_space();
}

AclKernelWriter *AclScopedKernelWriter::operator->()
{
    return _writer;
}

const AclKernelWriter *AclScopedKernelWriter::operator->() const
{
    return _writer;
}

AclKernelWriter *AclScopedKernelWriter::writer()
{
    return _writer;
}

const AclKernelWriter *AclScopedKernelWriter::writer() const
{
    return _writer;
}
