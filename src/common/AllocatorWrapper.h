/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_COMMON_ALLOCATORWRAPPER_H
#define SRC_COMMON_ALLOCATORWRAPPER_H

#include "arm_compute/AclTypes.h"

namespace arm_compute
{
/** Default malloc allocator implementation */
class AllocatorWrapper final
{
public:
    /** Default Constructor
     *
     * @param[in] backing_allocator Backing memory allocator to be used
     */
    AllocatorWrapper(const AclAllocator &backing_allocator) noexcept;
    AllocatorWrapper(const AllocatorWrapper &) noexcept = default;
    AllocatorWrapper(AllocatorWrapper &&) noexcept      = default;
    AllocatorWrapper &operator=(const AllocatorWrapper &) noexcept = delete;
    AllocatorWrapper &operator=(AllocatorWrapper &&other) noexcept = default;
    /** Allocate a chunk of memory of a given size in bytes
     *
     * @param[in] size Size of memory to allocate in bytes
     *
     * @return A pointer to the allocated memory if successful else nullptr
     */
    void *alloc(size_t size);
    /** Free an allocated memory block
     *
     * @param[in] ptr Pointer to allocated memory
     */
    void free(void *ptr);
    /** Allocate a chunk of memory of a given size in bytes,
     *  while honoring a given alignment requirement
     *
     * @param[in] size      Size of memory to allocate in bytes
     * @param[in] alignment Alignment requirements
     *
     * @return A pointer to the allocated memory if successful else nullptr
     */
    void *aligned_alloc(size_t size, size_t alignment);
    /** Free an aligned memory block
     *
     * @param[in] ptr Pointer to the memory to release
     */
    void aligned_free(void *ptr);
    /** Set user data to be used by the allocator
     *
     * @param[in] user_data User data to be used by the allocator
     */
    void set_user_data(void *user_data);

private:
    AclAllocator _backing_allocator;
};
} // namespace arm_compute

#endif /* SRC_COMMON_ALLOCATORWRAPPER_H */