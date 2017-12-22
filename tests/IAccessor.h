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
#ifndef __ARM_COMPUTE_TEST_IACCESSOR_H__
#define __ARM_COMPUTE_TEST_IACCESSOR_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
/** Common interface to provide information and access to tensor like
 * structures.
 */
class IAccessor
{
public:
    /** Virtual destructor. */
    virtual ~IAccessor() = default;

    /** Shape of the tensor. */
    virtual TensorShape shape() const = 0;

    /** Size of each element in the tensor in bytes. */
    virtual size_t element_size() const = 0;

    /** Total size of the tensor in bytes. */
    virtual size_t size() const = 0;

    /** Image format of the tensor. */
    virtual Format format() const = 0;

    /** Data type of the tensor. */
    virtual DataType data_type() const = 0;

    /** Number of channels of the tensor. */
    virtual int num_channels() const = 0;

    /** Number of elements of the tensor. */
    virtual int num_elements() const = 0;

    /** Available padding around the tensor. */
    virtual PaddingSize padding() const = 0;

    /** Number of bits for the fractional part. */
    virtual int fixed_point_position() const = 0;

    /** Quantization info in case of asymmetric quantized type */
    virtual QuantizationInfo quantization_info() const = 0;

    /** Read only access to the specified element.
     *
     * @param[in] coord Coordinates of the desired element.
     *
     * @return A pointer to the desired element.
     */
    virtual const void *operator()(const Coordinates &coord) const = 0;

    /** Access to the specified element.
     *
     * @param[in] coord Coordinates of the desired element.
     *
     * @return A pointer to the desired element.
     */
    virtual void *operator()(const Coordinates &coord) = 0;
};
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_IACCESSOR_H__ */
