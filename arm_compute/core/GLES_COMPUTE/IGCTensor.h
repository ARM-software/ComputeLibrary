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
#ifndef __ARM_COMPUTE_IGCTENSOR_H__
#define __ARM_COMPUTE_IGCTENSOR_H__

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/ITensor.h"

#include <cstdint>

namespace arm_compute
{
/** Interface for GLES Compute tensor */
class IGCTensor : public ITensor
{
public:
    /** Default constructor. */
    IGCTensor();

    /** Prevent instances of this class from being copied (As this class contains pointers). */
    IGCTensor(const IGCTensor &) = delete;

    /** Prevent instances of this class from being copy assigned (As this class contains pointers). */
    IGCTensor &operator=(const IGCTensor &) = delete;

    /** Allow instances of this class to be moved */
    IGCTensor(IGCTensor &&) = default;

    /** Allow instances of this class to be moved */
    IGCTensor &operator=(IGCTensor &&) = default;

    /** Virtual destructor */
    virtual ~IGCTensor() = default;

    /** Map on an allocated buffer.
     *
     * @param[in] blocking (Optional) If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     */
    void map(bool blocking = true);
    /** Unmap an allocated and mapped buffer.
     */
    void unmap();
    /** Clear the contents of the tensor synchronously.
     */
    void clear();

    // Inherited methods overridden:
    uint8_t *buffer() const override;
    /** Interface to be implemented by the child class to return the tensor's gles compute buffer id.
      *
      * @return A SSBO buffer id.
     */
    virtual GLuint gc_buffer() const = 0;

protected:
    /** Method to be implemented by the child class to map the SSBO.
     *
     * @param[in] blocking If true, then the mapping will be ready to use by the time
     *                     this method returns, else it is the caller's responsibility
     *                     to flush the queue and wait for the mapping operation to have completed before using the returned mapping pointer.
     */
    virtual uint8_t *do_map(bool blocking) = 0;
    /** Method to be implemented by the child class to unmap the SSBO.
     *
     * @note This method simply enqueues the unmap operation, it is the caller's responsibility to flush the queue and make sure the unmap is finished before
     *       the memory is accessed by the device.
     */
    virtual void do_unmap() = 0;

private:
    uint8_t *_mapping;
};

using IGCImage = IGCTensor;
}
#endif /*__ARM_COMPUTE_IGCTENSOR_H__ */
