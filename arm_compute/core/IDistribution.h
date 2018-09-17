/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_IDISTRIBUTION_H__
#define __ARM_COMPUTE_IDISTRIBUTION_H__

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
/** Interface for distribution objects */
class IDistribution
{
public:
    /** Default virtual destructor */
    virtual ~IDistribution() = default;
    /** Returns the dimensions of the distribution.
     *
     * @note  This is fixed to 1-dimensional distribution for now.
     * @return Dimensions of the distribution.
     */
    virtual size_t dimensions() const = 0;
    /** Returns the total size in bytes of the distribution.
     *
     * @return Total size of the distribution in bytes.
     */
    virtual size_t size() const = 0;
    /** Returns a pointer to the start of the distribution.
     * Other elements of the array can be accessed using buffer()[idx] for 0 <= idx < num_bins()
     *
     * @return Pointer to the start of the distribution.
     */
    virtual uint32_t *buffer() const = 0;
    /** Clears the distribution by setting every element to zero. */
    void clear() const;
};
}
#endif /* __ARM_COMPUTE_IDISTRIBUTION_H__ */
