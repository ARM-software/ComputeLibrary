/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_IARRAYACCESSOR_H__
#define __ARM_COMPUTE_TEST_IARRAYACCESSOR_H__

namespace arm_compute
{
namespace test
{
/** Common interface to provide information and access to array like
 * structures.
 */
template <typename T>
class IArrayAccessor
{
public:
    using value_type = T;

    /** Virtual destructor. */
    virtual ~IArrayAccessor() = default;

    /** Number of elements of the tensor. */
    virtual size_t num_values() const = 0;

    /** Access to the buffer.
     *
     * @return A pointer to the first element in the buffer.
     */
    virtual T *buffer() = 0;

    /** Resize array.
     *
     * @param[in] num The new array size in number of elements
     */
    virtual void resize(size_t num) = 0;

    /** Reference to the element of the array located at the given index
     *
     * @param[in] index Index of the element
     *
     * @return A reference to the element of the array located at the given index.
     */
    virtual T &at(size_t index) const = 0;
};
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_IARRAYACCESSOR_H__ */
