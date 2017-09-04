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
#ifndef __ARM_COMPUTE_CLHOG_H__
#define __ARM_COMPUTE_CLHOG_H__

#include "arm_compute/core/CL/ICLHOG.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/Types.h"

#include <cstdint>

namespace arm_compute
{
/** OpenCL implementation of HOG data-object */
class CLHOG : public ICLHOG
{
public:
    /** Default constructor */
    CLHOG();
    /** Allocate the HOG descriptor using the given HOG's metadata
     *
     * @param[in] input HOG's metadata used to allocate the HOG descriptor
     */
    void init(const HOGInfo &input);

    /** Enqueue a map operation of the allocated buffer.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed.
     */
    void map(bool blocking = true);
    using ICLHOG::map;

    /** Enqueue an unmap operation of the allocated and mapped buffer.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     */
    void unmap();
    using ICLHOG::unmap;

    // Inherited method overridden:
    void              free() override;
    const HOGInfo    *info() const override;
    const cl::Buffer &cl_buffer() const override;

protected:
    // Inherited methods overridden:
    uint8_t *do_map(cl::CommandQueue &q, bool blocking) override;
    void do_unmap(cl::CommandQueue &q) override;

private:
    HOGInfo    _info;
    cl::Buffer _buffer;
};
}
#endif /* __ARM_COMPUTE_CLHOG_H__ */
