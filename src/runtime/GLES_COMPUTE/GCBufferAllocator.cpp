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
#include "arm_compute/runtime/GLES_COMPUTE/GCBufferAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensorAllocator.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"

#include <cstddef>

namespace arm_compute
{
void *GCBufferAllocator::allocate(size_t size, size_t alignment)
{
    ARM_COMPUTE_UNUSED(alignment);
    auto *gl_buffer = new GLBufferWrapper();
    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_SHADER_STORAGE_BUFFER, gl_buffer->_ssbo_name));
    ARM_COMPUTE_GL_CHECK(glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(size), nullptr, GL_STATIC_DRAW));
    ARM_COMPUTE_GL_CHECK(glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0));

    return reinterpret_cast<void *>(gl_buffer);
}

void GCBufferAllocator::free(void *ptr)
{
    ARM_COMPUTE_ERROR_ON(ptr == nullptr);
    auto *gl_buffer = reinterpret_cast<GLBufferWrapper *>(ptr);
    delete gl_buffer;
}

std::unique_ptr<IMemoryRegion> GCBufferAllocator::make_region(size_t size, size_t alignment)
{
    ARM_COMPUTE_UNUSED(size, alignment);
    return nullptr;
}
} // namespace arm_compute
