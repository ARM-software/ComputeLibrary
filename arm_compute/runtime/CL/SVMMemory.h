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
#ifndef __ARM_COMPUTE_SVMMEMORY_H__
#define __ARM_COMPUTE_SVMMEMORY_H__

namespace arm_compute
{
    class SVMMemory final
    {
    public:
        SVMMemory() = default;
        SVMMemory(void *ptr, bool fine_grain)
            : _ptr(ptr), _fine_grain(fine_grain), _size(0)
        {
        }
        void *ptr() const
        {
            return _ptr;
        }
        bool fine_grain() const
        {
            return _fine_grain;
        }
        size_t size() const
        {
            return _size;
        }
        void *allocate(cl_context context, size_t size, cl_svm_mem_flags flags, cl_uint alignment);
    private:
        void *_ptr{ nullptr };
        bool   _fine_grain{ false };
        size_t _size{ 0 };
    };
}
#endif /* __ARM_COMPUTE_SVMMEMORY_H__ */
