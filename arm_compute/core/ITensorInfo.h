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
#ifndef __ARM_COMPUTE_ITENSORINFO_H__
#define __ARM_COMPUTE_ITENSORINFO_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ICloneable.h"

#include <cstddef>

namespace arm_compute
{
/** Store the tensor's metadata */
class ITensorInfo : public misc::ICloneable<ITensorInfo>
{
public:
    /** Default virtual destructor */
    virtual ~ITensorInfo() = default;
    /** Set the data type to the specified value.
     *
     * @warning This resets the format to UNKNOWN.
     *
     * @param[in] data_type The new data type.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_data_type(DataType data_type) = 0;
    /** Set the number of channels to the specified value.
     *
     * @warning This resets the format to UNKNOWN.
     *
     * @param[in] num_channels New number of channels.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_num_channels(int num_channels) = 0;
    /** Set the format of an already initialized tensor.
     *
     * @note If the data type has already been configured (i.e. not UNKNOWN) it
     * must match the new format. If data type hasn't been configured it will
     * be based on the format.
     *
     * @param[in] format Single-plane format of the tensor.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_format(Format format) = 0;
    /** Set the shape of an already initialized tensor.
     *
     * @warning Changing the shape requires to recompute the strides and is
     * therefore only possible if the tensor hasn't been allocated yet.
     *
     * @param[in] shape New tensor shape.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_tensor_shape(TensorShape shape) = 0;
    /** Set the fixed point position to the specified value
     *
     * @warning The fixed point position must be set once the data type has been configured
     *
     * @param[in] fixed_point_position The new fixed point position
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_fixed_point_position(int fixed_point_position) = 0;
    /** Set the quantization settings (scale and offset) of the tensor.
     *
     * @param[in] quantization_info QuantizationInfo containing the scale and offset
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_quantization_info(QuantizationInfo quantization_info) = 0;
    /** Resets the padding settings of the tensor.
    *
    * @return Reference to this ITensorInfo object
    */
    virtual ITensorInfo &reset_padding() = 0;
    /** Update the offset to the first element and the strides to automatically computed values.
     *
     * @note The padding used by this method is really conservative so that the tensor can be used for most functions.
     *
     * @return True if the strides or the offset to the first element have changed.
     */
    virtual bool auto_padding() = 0;
    /** Update the offset to the first element, the strides and the total size.
     *
     * @note This function can only increase the offset, strides and total size.
     *
     * @param[in] padding Padding around the XY plane in number of elements.
     *
     * @return True if the strides, offset and total size have changed.
     */
    virtual bool extend_padding(const PaddingSize &padding) = 0;
    /** Return the size of the requested dimension
     *
     * @param[in] index Index of the dimension
     *
     * @return Dimension of the requested dimension
     */
    virtual size_t dimension(size_t index) const = 0;
    /** The strides in bytes for accessing each dimension of the tensor
     *
     * @return Strides in bytes for each tensor dimension
     */
    virtual const Strides &strides_in_bytes() const = 0;
    /** The offset from the beginning of the memory allocation to the first element of the tensor.
     *  This can be used to access efficiently elements in a 2D tensor
     *
     * @return The offset in bytes to access the first element of the tensor.
     */
    virtual size_t offset_first_element_in_bytes() const = 0;
    /** The offset in bytes from the beginning of the memory allocation to access the element at position (x, y, z ...)
     *
     * @param[in] pos Vector with the coordinates of the element to access.
     *                The size of this vector must be equal to the number of dimensions of the tensor
     *
     * @return Offset in bytes from the beginning of the memory allocation to access the element (x, y, z, ...)
     */
    virtual size_t offset_element_in_bytes(const Coordinates &pos) const = 0;
    /** Fixed point position used when the tensor data type is QS8 or QS16
     *
     * @return The fixed point position that expresses the number of bits for the fractional part of the number
     */
    virtual int fixed_point_position() const = 0;
    /** Element size in bytes calculated as data_size() * num_channels()
     *
     * @return The size of one element in bytes
     */
    virtual size_t element_size() const = 0;
    /** The number of dimensions of the tensor (rank)
     *
     * @return The number of dimensions of the tensor (rank)
     */
    virtual size_t num_dimensions() const = 0;
    /** The number of channels for each tensor element
     *
     * @return The number of channels for each tensor element
     */
    virtual size_t num_channels() const = 0;
    /** Size for each dimension of the tensor
     *
     * @return A vector with the size for each dimension of the tensor
     */
    virtual const TensorShape &tensor_shape() const = 0;
    /** Data type used for each element of the tensor
     *
     * @return Tensor data type
     */
    virtual DataType data_type() const = 0;
    /** Colour format of the image
     *
     * @return Colour format of the image
     */
    virtual Format format() const = 0;
    /** Returns the total size of the tensor in bytes.
     *
     * @return Total size of the tensor in bytes.
     */
    virtual size_t total_size() const = 0;
    /** Padding of tensor.
     *
     * @return Padding.
     */
    virtual PaddingSize padding() const = 0;
    /** Checks if the tensor has been allocated with padding or not.
     *
     * @return True if padding is allocated in the tensor, otherwise false.
     */
    virtual bool has_padding() const = 0;
    /** Flag indicating whether the size of the tensor can be changed.
     *
     * @return True if the tensor size can be changed.
     */
    virtual bool is_resizable() const = 0;
    /** Set the flag whether the tensor size can be changed.
     *
     * @param[in] is_resizable Flag that marks the tensor if it can be changed or not.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_is_resizable(bool is_resizable) = 0;
    /** Valid region of the tensor. All elements in the valid region have defined values, i.e. are not undefined.
     *
     * @return The valid region.
     */
    virtual ValidRegion valid_region() const = 0;
    /** Set the valid region of the tensor.
     *
     * @param[in] valid_region Valid region to set.
     */
    virtual void set_valid_region(ValidRegion valid_region) = 0;

    /** Get the quantization settings (scale and offset) of the tensor.
    *
    * @return A QuantizationInfo containing the scale and offset.
    */
    virtual QuantizationInfo quantization_info() const = 0;
};
}
#endif /*__ARM_COMPUTE_TENSORINFO_H__ */
