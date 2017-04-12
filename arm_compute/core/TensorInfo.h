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
#ifndef __ARM_COMPUTE_TENSORINFO_H__
#define __ARM_COMPUTE_TENSORINFO_H__

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"

#include <cstddef>

namespace arm_compute
{
class HOGInfo;

/** Store the tensor's metadata */
class TensorInfo
{
public:
    /** Default constructor */
    TensorInfo();
    /** Default destructor */
    virtual ~TensorInfo() = default;
    /** Allow instances of this class to be copy constructed */
    TensorInfo(const TensorInfo &) = default;
    /** Allow instances of this class to be copied */
    TensorInfo &operator=(const TensorInfo &) = default;
    /** Allow instances of this class to be move constructed */
    TensorInfo(TensorInfo &&) = default;
    /** Allow instances of this class to be moved */
    TensorInfo &operator=(TensorInfo &&) = default;
    /** 2D tensor constructor
     *
     * @param[in] width  Width of the 2D tensor
     * @param[in] height Height of the 2D tensor
     * @param[in] format Single plane format of the tensor.
     */
    TensorInfo(unsigned int width, unsigned int height, Format format);
    /** Constructor
     *
     * @param[in] tensor_shape It specifies the size for each dimension of the tensor in number of elements.
     * @param[in] format       Single plane format of the tensor.
     */
    TensorInfo(const TensorShape &tensor_shape, Format format);
    /** Constructor
     *
     * @param[in] tensor_shape    It specifies the size for each dimension of the tensor in number of elements.
     * @param[in] num_channels    It indicates the number of channels for each tensor element
     * @param[in] data_type       Data type to use for each tensor element
     * @param[in] fixed_point_pos (Optional) It specifies the fixed point position when the tensor data type is INT8, INT16 or INT32. (Default = 0)
                                  If 0, calculations are performed in integer math
     */
    TensorInfo(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, size_t fixed_point_pos = 0);
    /** Constructor
     *
     * @param[in] hog_info HOG's metadata used to allocate normalized HOG space
     * @param[in] width    Width of the 2D tensor where the HOG descriptor will be computed on
     * @param[in] height   Height of the 2D tensor where the HOG descriptor will be computed on
     */
    TensorInfo(const HOGInfo &hog_info, unsigned int width, unsigned int height);
    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] tensor_shape Size for each dimension of the tensor in number of elements.
     * @param[in] format       Single plane format of the tensor.
     */
    void init(const TensorShape &tensor_shape, Format format);
    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] tensor_shape                  Size for each dimension of the tensor in number of elements.
     * @param[in] format                        Single plane format of the tensor.
     * @param[in] strides_in_bytes              Stride in bytes for accessing each dimension of the tensor.
     * @param[in] offset_first_element_in_bytes Offset in bytes from the beginning of memory allocation to access the first element.
     * @param[in] total_size_in_bytes           Size in bytes of the memory allocation (including the offset to the first element).
     */
    void init(const TensorShape &tensor_shape, Format format, const Strides &strides_in_bytes, size_t offset_first_element_in_bytes, size_t total_size_in_bytes);
    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] tensor_shape    Size for each dimension of the tensor in number of elements.
     * @param[in] num_channels    Desired number of channels for each tensor element.
     * @param[in] data_type       Data type to use for each tensor element.
     * @param[in] fixed_point_pos (Optional) Fixed point position when the tensor data type is INT8, INT16 or INT32 (default = 0).
     *                            If 0, calculations are performed in integer arithmetic.
     */
    void init(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, size_t fixed_point_pos = 0);
    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] tensor_shape                  Size for each dimension of the tensor in number of elements.
     * @param[in] num_channels                  Desired number of channels for each tensor element.
     * @param[in] data_type                     Data type to use for each tensor element.
     * @param[in] strides_in_bytes              Stride in bytes for accessing each dimension of the tensor.
     * @param[in] offset_first_element_in_bytes Offset in bytes from the beginning of memory allocation to access the first element.
     * @param[in] total_size_in_bytes           Size in bytes of the memory allocation (including the offset to the first element).
     * @param[in] fixed_point_pos               (Optional) Fixed point position when the tensor data type is INT8, INT16 or INT32 (default = 0).
     *                                          If 0, calculations are performed in integer arithmetic.
     */
    void init(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, const Strides &strides_in_bytes, size_t offset_first_element_in_bytes,
              size_t total_size_in_bytes, size_t fixed_point_pos = 0);
    /** Initialize the metadata structure for the given HOG's metadata
     *
     * @param[in] hog_info HOG's metadata used to allocate normalized HOG space
     * @param[in] width    Width of the 2D tensor where the HOG descriptor will be computed on
     * @param[in] height   Height of the 2D tensor where the HOG descriptor will be computed on
     */
    void init(const HOGInfo &hog_info, unsigned int width, unsigned int height);
    /** Initialize the metadata structure for the given tensor shape and single-plane format, (Padding is automatically calculated)
     *
     * @note The padding used by this method is really conservative so that the tensor can be used for most functions.
     *
     * @param[in] tensor_shape It specifies the size for each dimension of the tensor in number of elements
     * @param[in] format       Single plane format of the image.
     *
     * @return Total allocation size including padding in bytes.
     */
    size_t init_auto_padding(const TensorShape &tensor_shape, Format format);
    /** Initialize the metadata structure for the given tensor shape, number of channels,
     *  data type and fixed point position. (Padding is automatically calculated)
     *
     * @note The padding used by this method is really conservative so that the tensor can be used for most functions.
     *
     * @param[in] tensor_shape    It specifies the size for each dimension of the tensor in number of elements
     * @param[in] num_channels    It indicates the number of channels for each tensor element
     * @param[in] data_type       Data type to use for each tensor element
     * @param[in] fixed_point_pos (Optional) It specifies the fixed point position when the tensor data type is INT8, INT16 or INT32. (Default = 0)
     *                            If 0, calculations are performed in integer math
     *
     * @return Total allocation size including padding in bytes.
     */
    size_t init_auto_padding(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, size_t fixed_point_pos = 0);
    /** Initialize the metadata structure for the given HOG's metadata
     *
     * @note init_auto_padding will be used for the tensor initialization.
     *
     * @param[in] hog_info HOG's metadata used to allocate normalized HOG space
     * @param[in] width    Width of the 2D tensor where the HOG descriptor will be computed on
     * @param[in] height   Height of the 2D tensor where the HOG descriptor will be computed on
     */
    size_t init_auto_padding(const HOGInfo &hog_info, unsigned int width, unsigned int height);
    /** Update the offset to the first element and the strides to automatically computed values.
     *
     * @note The padding used by this method is really conservative so that the tensor can be used for most functions.
     *
     * @return True if the strides or the offset to the first element have changed.
     */
    bool auto_padding();
    /** Update the offset to the first element, the strides and the total size.
     *
     * @note This function can only increase the offset, strides and total size.
     *
     * @param[in] padding Padding around the XY plane in number of elements.
     *
     * @return True if the strides, offset and total size have changed.
     */
    bool extend_padding(const PaddingSize &padding);
    /** Set the format of an already initialized tensor.
     *
     * @note The passed format must be compatible with the existing number of channels and data type of the tensor.
     *
     * @param[in] format Single-plane format of the tensor.
     */
    void set_format(Format format);
    /** Return the size of the requested dimension
     *
     * @param[in] index Index of the dimension
     *
     * @return Dimension of the requested dimension
     */
    size_t dimension(size_t index) const
    {
        return _tensor_shape[index];
    }
    /** The strides in bytes for accessing each dimension of the tensor
     *
     * @return Strides in bytes for each tensor dimension
     */
    const Strides &strides_in_bytes() const
    {
        return _strides_in_bytes;
    }
    /** The offset from the beginning of the memory allocation to the first element of the tensor.
     *  This can be used to access efficiently elements in a 2D tensor
     *
     * @return The offset in bytes to access the first element of the tensor.
     */
    size_t offset_first_element_in_bytes() const
    {
        return _offset_first_element_in_bytes;
    }
    /** The offset in bytes from the beginning of the memory allocation to access the element at position (x, y, z ...)
     *
     * @param[in] pos Vector with the coordinates of the element to access.
     *                The size of this vector must be equal to the number of dimensions of the tensor
     *
     * @return Offset in bytes from the beginning of the memory allocation to access the element (x, y, z, ...)
     */
    size_t offset_element_in_bytes(const Coordinates &pos) const;
    /** Fixed point position used when the tensor data type is S8, S16 or S32.
     *
     * @return The fixed point position
     */
    size_t fixed_point_pos() const
    {
        return _fixed_point_pos;
    }
    /** Element size in bytes calculated as data_size() * num_channels
     *
     * @return The size of one element in bytes
     */
    size_t element_size() const
    {
        return data_size_from_type(_data_type) * _num_channels;
    }
    /** The number of dimensions of the tensor (rank)
     *
     * @return The number of dimensions of the tensor (rank)
     */
    size_t num_dimensions() const
    {
        return _tensor_shape.num_dimensions();
    }
    /** The number of channels for each tensor element
     *
     * @return The number of channels for each tensor element
     */
    size_t num_channels() const
    {
        return _num_channels;
    }
    /** Size for each dimension of the tensor
     *
     * @return A vector with the size for each dimension of the tensor
     */
    const TensorShape &tensor_shape() const
    {
        return _tensor_shape;
    }
    /** Data type used for each element of the tensor
     *
     * @return Tensor data type
     */
    DataType data_type() const
    {
        return _data_type;
    }
    /** Colour format of the image
     *
     * @return Colour format of the image
     */
    Format format() const
    {
        return _format;
    }
    /** Returns the total size of the tensor in bytes.
     *
     * @return Total size of the tensor in bytes.
     */
    size_t total_size() const
    {
        return _total_size;
    }
    /** Padding of tensor.
     *
     * @return Padding.
     */
    PaddingSize padding() const
    {
        return _padding;
    }
    /** Checks if the tensor has been allocated with padding or not.
     *
     * @return True if padding is allocated in the tensor, otherwise false.
     */
    bool has_padding() const
    {
        return !_padding.empty();
    }
    /** Flag indicating whether the size of the tensor can be changed.
     *
     * @return True if the tensor size can be changed.
     */
    bool is_resizable() const
    {
        return _is_resizable;
    }
    /** Set the flag whether the tensor size can be changed. */
    void set_is_resizable(bool is_resizable)
    {
        _is_resizable = is_resizable;
    }
    /** Valid region of the tensor. All elements in the valid region have defined values, i.e. are not undefined.
     *
     * @return The valid region.
     */
    ValidRegion valid_region() const
    {
        return _valid_region;
    }
    /** Set the valid region of the tensor. */
    void set_valid_region(ValidRegion valid_region)
    {
        _valid_region = std::move(valid_region);
    }

private:
    /** Calculates strides, offset and total size resulting from the specified padding around the XY plane.
     *
     * @param[in] padding Padding around the XY plane in elements.
     */
    std::tuple<Strides, size_t, size_t> calculate_padding_requirements(const PaddingSize &padding);

    size_t      _total_size;
    size_t      _fixed_point_pos;
    size_t      _offset_first_element_in_bytes;
    Strides     _strides_in_bytes;
    size_t      _num_channels;
    TensorShape _tensor_shape;
    DataType    _data_type;
    Format      _format;
    bool        _is_resizable;
    ValidRegion _valid_region;
    PaddingSize _padding;
};
}
#endif /*__ARM_COMPUTE_TENSORINFO_H__ */
