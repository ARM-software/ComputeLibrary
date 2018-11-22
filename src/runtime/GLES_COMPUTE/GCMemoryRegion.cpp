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
#include "arm_compute/runtime/GLES_COMPUTE/GCMemoryRegion.h"

#include "arm_compute/core/Error.h"

namespace arm_compute
{
IGCMemoryRegion::IGCMemoryRegion(size_t size)
    : IMemoryRegion(size), _mapping(nullptr), _ssbo_name(0)
{
}

const GLuint &IGCMemoryRegion::gc_ssbo_name() const
{
    return _ssbo_name;
}

void *IGCMemoryRegion::buffer()
{
    return _mapping;
}

void *IGCMemoryRegion::buffer() const
{
    return _mapping;
}

GCBufferMemoryRegion::GCBufferMemoryRegion(size_t size)
    : IGCMemoryRegion(size)
{
    ARM_COMPUTE_GL_CHECK(glGenBuffers(1, &_ssbo_name));
    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssbo_name));
    ARM_COMPUTE_GL_CHECK(glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(size), nullptr, GL_STATIC_DRAW));
    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0));
}

GCBufferMemoryRegion::~GCBufferMemoryRegion()
{
    ARM_COMPUTE_GL_CHECK(glDeleteBuffers(1, &_ssbo_name));
}

void *GCBufferMemoryRegion::ptr()
{
    return nullptr;
}

void *GCBufferMemoryRegion::map(bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_mapping != nullptr);
    ARM_COMPUTE_UNUSED(blocking);

    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssbo_name));
    void *p  = ARM_COMPUTE_GL_CHECK(glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(size()), GL_MAP_READ_BIT | GL_MAP_WRITE_BIT));
    _mapping = reinterpret_cast<uint8_t *>(p);

    return _mapping;
}

void GCBufferMemoryRegion::unmap()
{
    ARM_COMPUTE_ERROR_ON(_mapping == nullptr);

    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssbo_name));
    ARM_COMPUTE_GL_CHECK(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER));
    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0));
    _mapping = nullptr;
}

std::unique_ptr<IMemoryRegion> GCBufferMemoryRegion::extract_subregion(size_t offset, size_t size)
{
    ARM_COMPUTE_UNUSED(offset, size);
    return nullptr;
}
} // namespace arm_compute