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
#ifndef __ARM_COMPUTE_CLDISTRIBUTION1D_H__
#define __ARM_COMPUTE_CLDISTRIBUTION1D_H__

#include "arm_compute/core/CL/ICLDistribution1D.h"
#include "arm_compute/core/CL/OpenCL.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
/** CLDistribution1D object class */
class CLDistribution1D : public ICLDistribution1D
{
public:
    /** Constructor: Creates a 1D CLDistribution of a consecutive interval [offset, offset + range - 1]
     *               defined by a start offset and valid range, divided equally into num_bins parts.
     *
     * @param[in] num_bins The number of bins the distribution is divided in.
     * @param[in] offset   The start of the values to use.
     * @param[in] range    The total number of the consecutive values of the distribution interval.
     */
    CLDistribution1D(size_t num_bins, int32_t offset, uint32_t range);
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLDistribution1D(const CLDistribution1D &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers). */
    CLDistribution1D &operator=(const CLDistribution1D &) = delete;
    /** Enqueue a map operation of the allocated buffer.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed.
     */
    void map(bool blocking = true);
    using ICLDistribution1D::map;
    /** Enqueue an unmap operation of the allocated and mapped buffer.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     */
    void unmap();
    using ICLDistribution1D::unmap;

    // Inherited methods overridden:
    cl::Buffer &cl_buffer() override;

protected:
    // Inherited methods overridden:
    uint32_t *do_map(cl::CommandQueue &q, bool blocking) override;
    void do_unmap(cl::CommandQueue &q) override;

private:
    cl::Buffer _mem;
};
}
#endif /* __ARM_COMPUTE_CLDISTRIBUTION1D_H__ */
