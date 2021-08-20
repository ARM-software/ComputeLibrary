/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_ITENSOR_H
#define ARM_COMPUTE_ITENSOR_H

#include "arm_compute/core/ITensorInfo.h"

#include <cstdint>

namespace arm_compute
{
class Coordinates;

/** Interface for CPU tensor */
class ITensor
{
public:
    /** Interface to be implemented by the child class to return the tensor's metadata
     *
     * @return A pointer to the tensor's metadata.
     */
    virtual ITensorInfo *info() const = 0;
    /** Interface to be implemented by the child class to return the tensor's metadata
     *
     * @return A pointer to the tensor's metadata.
     */
    virtual ITensorInfo *info() = 0;
    /** Default virtual destructor */
    virtual ~ITensor() = default;
    /** Interface to be implemented by the child class to return a pointer to CPU memory
     *
     * @return A CPU pointer to the beginning of the image's allocation.
     */
    virtual uint8_t *buffer() const = 0;

    /** Return a pointer to the element at the passed coordinates
     *
     * @param[in] id Coordinates of the element
     *
     * @return Pointer to the requested element
     */
    inline uint8_t *ptr_to_element(const Coordinates &id) const
    {
        return buffer() + info()->offset_element_in_bytes(id);
    }

    /** Copy the content of another tensor.
     *
     * @note The number of dimensions of the source tensor must be less or equal to those of the destination tensor.
     *
     * @note All dimensions of the destination tensor must be greater or equal to the source tensor ones.
     *
     * @note num_channels() and element_size() of both tensors must match.
     *
     * @param[in] src Source tensor to copy from.
     */
    void copy_from(const ITensor &src);

    /** Print a tensor to a given stream using user defined formatting information
     *
     * @param s      Output stream
     * @param io_fmt Format information
     */
    void print(std::ostream &s, IOFormatInfo io_fmt = IOFormatInfo()) const;
    /** Flags if the tensor is used or not
     *
     * @return True if it is used else false
     */
    bool is_used() const;
    /** Marks a tensor as unused */
    void mark_as_unused() const;
    /** Marks a tensor as used */
    void mark_as_used() const;

private:
    mutable bool _is_used = { true }; /**< Flag that marks if the tensor is used or not */
};

using IImage = ITensor;
}
#endif /*ARM_COMPUTE_ITENSOR_H */
