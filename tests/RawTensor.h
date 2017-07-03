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

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace arm_compute
{
namespace test
{
/** Simple tensor object that stores elements in a consecutive chunk of memory.
 *
 * It can be created by either loading an image from a file which also
 * initialises the content of the tensor or by explcitly specifying the size.
 * The latter leaves the content uninitialised.
 *
 * Furthermore, the class provides methods to convert the tensor's values into
 * different image format.
 */
class RawTensor final
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

    /** Create a deep copy of the given @p tensor.
     *
     * @param[in] tensor To be copied tensor.
     */
    RawTensor &operator     =(RawTensor tensor);
    RawTensor(RawTensor &&) = default;
    ~RawTensor()            = default;

    using BufferType = uint8_t;
    using Buffer     = std::unique_ptr<BufferType[]>;

    /** Return value at @p offset in the buffer.
     *
     * @param[in] offset Offset within the buffer.
     */
    BufferType &operator[](size_t offset);

    /** Return constant value at @p offset in the buffer.
     *
     * @param[in] offset Offset within the buffer.
     */
    const BufferType &operator[](size_t offset) const;

    /** Shape of the tensor. */
    TensorShape shape() const;

    /** Size of each element in the tensor in bytes. */
    size_t element_size() const;

    /** Total size of the tensor in bytes. */
    size_t size() const;

    /** Image format of the tensor. */
    Format format() const;

    /** Data type of the tensor. */
    DataType data_type() const;

    /** Number of channels of the tensor. */
    int num_channels() const;

    /** Number of elements of the tensor. */
    int num_elements() const;

    /** The number of bits for the fractional part of the fixed point numbers. */
    int fixed_point_position() const;

    /** Constant pointer to the underlying buffer. */
    const BufferType *data() const;

    /** Pointer to the underlying buffer. */
    BufferType *data();

    /** Read only access to the specified element.
     *
     * @param[in] coord Coordinates of the desired element.
     *
     * @return A pointer to the desired element.
     */
    const BufferType *operator()(const Coordinates &coord) const;

    /** Access to the specified element.
     *
     * @param[in] coord Coordinates of the desired element.
     *
     * @return A pointer to the desired element.
     */
    BufferType *operator()(const Coordinates &coord);

    /** Swaps the content of the provided tensors.
     *
     * @param[in, out] tensor1 Tensor to be swapped.
     * @param[in, out] tensor2 Tensor to be swapped.
     */
    friend void swap(RawTensor &tensor1, RawTensor &tensor2);

private:
    Buffer      _buffer{ nullptr };
    TensorShape _shape{};
    Format      _format{ Format::UNKNOWN };
    DataType    _data_type{ DataType::UNKNOWN };
    int         _num_channels{ 0 };
    int         _fixed_point_position{ 0 };
};
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_RAW_TENSOR_H__ */
