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
#ifndef __ARM_COMPUTE_TEST_RAW_TENSOR_H__
#define __ARM_COMPUTE_TEST_RAW_TENSOR_H__

#include "SimpleTensor.h"

namespace arm_compute
{
namespace test
{
/** Subclass of SimpleTensor using uint8_t as value type.
 *
 * Access operations (except for operator[]) will be based on the data type to
 * copy the right number of elements.
 */
class RawTensor : public SimpleTensor<uint8_t>
{
public:
    /** Create an uninitialised tensor of the given @p shape and @p format.
     *
     * @param[in] shape                Shape of the new raw tensor.
     * @param[in] format               Format of the new raw tensor.
     * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers
     */
    RawTensor(TensorShape shape, Format format, int fixed_point_position = 0);

    /** Create an uninitialised tensor of the given @p shape and @p data type.
     *
     * @param[in] shape                Shape of the new raw tensor.
     * @param[in] data_type            Data type of the new raw tensor.
     * @param[in] num_channels         (Optional) Number of channels (default = 1).
     * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers (default = 0).
     */
    RawTensor(TensorShape shape, DataType data_type, int num_channels = 1, int fixed_point_position = 0);

    /** Create a deep copy of the given @p tensor.
     *
     * @param[in] tensor To be copied tensor.
     */
    RawTensor(const RawTensor &tensor);

    RawTensor &operator     =(RawTensor tensor);
    RawTensor(RawTensor &&) = default;
    ~RawTensor()            = default;

    /** Read only access to the specified element.
     *
     * @param[in] coord Coordinates of the desired element.
     *
     * @return A pointer to the desired element.
     */
    const void *operator()(const Coordinates &coord) const override;

    /** Access to the specified element.
     *
     * @param[in] coord Coordinates of the desired element.
     *
     * @return A pointer to the desired element.
     */
    void *operator()(const Coordinates &coord) override;
};
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_RAW_TENSOR_H__ */
