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

#include "arm_compute/core/ITensorInfo.h"

#include "ITensorInfo.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
class HOGInfo;

/** Store the tensor's metadata */
class TensorInfo final : public ITensorInfo
{
public:
    /** Default constructor */
    TensorInfo();
    /** Default destructor */
    ~TensorInfo() = default;
    /** Allow instances of this class to be copy constructed */
    TensorInfo(const ITensorInfo &info);
    /** Allow instances of this class to be copy constructed */
    TensorInfo(const TensorInfo &) = default;
    /** Allow instances of this class to be copied */
    TensorInfo &operator=(const TensorInfo &) = default;
    /** Allow instances of this class to be move constructed */
    TensorInfo(TensorInfo &&) = default;
    /** Allow instances of this class to be moved */
    TensorInfo &operator=(TensorInfo &&) = default;

    /** Construct a tensor info with a format.
     *
     * Can be used for automatic derivation of the shape by the function.
     *
     * @param[in] format Format of the tensor.
     */
    TensorInfo(Format format);

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

    /** Construct a tensor info with a data type and number of channels.
     *
     * Can be used for automatic derivation of the shape by the function.
     *
     * @param[in] num_channels         It indicates the number of channels for each tensor element
     * @param[in] data_type            Data type to use for each tensor element
     * @param[in] fixed_point_position (Optional) It specifies the fixed point position when the tensor data type is QS8, QS16 or QS32.
     */
    TensorInfo(size_t num_channels, DataType data_type, size_t fixed_point_position = 0);

    /** Constructor
     *
     * @param[in] tensor_shape         It specifies the size for each dimension of the tensor in number of elements.
     * @param[in] num_channels         It indicates the number of channels for each tensor element
     * @param[in] data_type            Data type to use for each tensor element
     * @param[in] fixed_point_position (Optional) Fixed point position that expresses the number of bits for the fractional part of the number when the tensor's data type is QS8 or QS16.
     */
    TensorInfo(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, int fixed_point_position = 0);

    /** Constructor
     *
     * @param[in] tensor_shape      It specifies the size for each dimension of the tensor in number of elements.
     * @param[in] num_channels      It indicates the number of channels for each tensor element
     * @param[in] data_type         Data type to use for each tensor element
     * @param[in] quantization_info The quantization settings for the tensor data.
     */
    TensorInfo(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, QuantizationInfo quantization_info);

    /** Constructor
     *
     * @param[in] hog_info HOG's metadata used to allocate normalized HOG space
     * @param[in] width    Width of the 2D tensor where the HOG descriptor will be computed on
     * @param[in] height   Height of the 2D tensor where the HOG descriptor will be computed on
     */
    TensorInfo(const HOGInfo &hog_info, unsigned int width, unsigned int height);

    /** Initialize the tensor info with just a format.
     *
     * Can be used for automatic derivation of the shape by the function.
     *
     * @param[in] format Single plane format of the tensor.
     */
    void init(Format format);

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

    /** Initialize the tensor info with just a format.
     *
     * Can be used for automatic derivation of the shape by the function.
     *
     * @param[in] num_channels         Desired number of channels for each tensor element.
     * @param[in] data_type            Data type to use for each tensor element.
     * @param[in] fixed_point_position (Optional) Fixed point position when the tensor data type is QS8, QS16 or QS32.
     */
    void init(size_t num_channels, DataType data_type, size_t fixed_point_position = 0);

    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] tensor_shape         Size for each dimension of the tensor in number of elements.
     * @param[in] num_channels         Desired number of channels for each tensor element.
     * @param[in] data_type            Data type to use for each tensor element.
     * @param[in] fixed_point_position (Optional) Fixed point position that expresses the number of bits for the fractional part of the number when the tensor's data type is QS8 or QS16.
     */
    void init(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, int fixed_point_position = 0);

    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] tensor_shape                  Size for each dimension of the tensor in number of elements.
     * @param[in] num_channels                  Desired number of channels for each tensor element.
     * @param[in] data_type                     Data type to use for each tensor element.
     * @param[in] strides_in_bytes              Stride in bytes for accessing each dimension of the tensor.
     * @param[in] offset_first_element_in_bytes Offset in bytes from the beginning of memory allocation to access the first element.
     * @param[in] total_size_in_bytes           Size in bytes of the memory allocation (including the offset to the first element).
     * @param[in] fixed_point_position          (Optional) Fixed point position that expresses the number of bits for the fractional part of the number when the tensor's data type is QS8 or QS16.
     */
    void init(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, const Strides &strides_in_bytes, size_t offset_first_element_in_bytes,
              size_t total_size_in_bytes, int fixed_point_position = 0);
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
     * @param[in] tensor_shape         It specifies the size for each dimension of the tensor in number of elements
     * @param[in] num_channels         It indicates the number of channels for each tensor element
     * @param[in] data_type            Data type to use for each tensor element
     * @param[in] fixed_point_position (Optional) Fixed point position that expresses the number of bits for the fractional part of the number when the tensor's data type is QS8 or QS16.
     *
     * @return Total allocation size including padding in bytes.
     */
    size_t init_auto_padding(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, int fixed_point_position = 0);
    /** Initialize the metadata structure for the given HOG's metadata
     *
     * @note init_auto_padding will be used for the tensor initialization.
     *
     * @param[in] hog_info HOG's metadata used to allocate normalized HOG space
     * @param[in] width    Width of the 2D tensor where the HOG descriptor will be computed on
     * @param[in] height   Height of the 2D tensor where the HOG descriptor will be computed on
     */
    size_t init_auto_padding(const HOGInfo &hog_info, unsigned int width, unsigned int height);

    // Inherited methods overridden:
    std::unique_ptr<ITensorInfo> clone() const override;
    ITensorInfo &set_data_type(DataType data_type) override;
    ITensorInfo &set_num_channels(int num_channels) override;
    ITensorInfo &set_format(Format format) override;
    ITensorInfo &set_tensor_shape(TensorShape shape) override;
    ITensorInfo &set_fixed_point_position(int fixed_point_position) override;
    ITensorInfo &set_quantization_info(QuantizationInfo quantization_info) override;
    ITensorInfo &reset_padding() override;
    bool         auto_padding() override;
    bool extend_padding(const PaddingSize &padding) override;
    size_t dimension(size_t index) const override
    {
        return _tensor_shape[index];
    }
    const Strides &strides_in_bytes() const override
    {
        return _strides_in_bytes;
    }
    size_t offset_first_element_in_bytes() const override
    {
        return _offset_first_element_in_bytes;
    }
    size_t offset_element_in_bytes(const Coordinates &pos) const override;
    int fixed_point_position() const override
    {
        return _fixed_point_position;
    }
    size_t element_size() const override
    {
        return data_size_from_type(_data_type) * _num_channels;
    }
    size_t num_dimensions() const override
    {
        return _tensor_shape.num_dimensions();
    }
    size_t num_channels() const override
    {
        return _num_channels;
    }
    const TensorShape &tensor_shape() const override
    {
        return _tensor_shape;
    }
    DataType data_type() const override
    {
        return _data_type;
    }
    Format format() const override
    {
        return _format;
    }
    size_t total_size() const override
    {
        return _total_size;
    }
    PaddingSize padding() const override
    {
        return _padding;
    }
    bool has_padding() const override
    {
        return !_padding.empty();
    }
    bool is_resizable() const override
    {
        return _is_resizable;
    }
    ITensorInfo &set_is_resizable(bool is_resizable) override
    {
        _is_resizable = is_resizable;
        return *this;
    }
    ValidRegion valid_region() const override
    {
        return _valid_region;
    }
    void set_valid_region(ValidRegion valid_region) override
    {
        _valid_region = std::move(valid_region);
    }
    QuantizationInfo quantization_info() const override
    {
        return _quantization_info;
    }

private:
    /** Calculates strides, offset and total size resulting from the specified padding around the XY plane.
     *
     * @param[in] padding Padding around the XY plane in elements.
     */
    std::tuple<Strides, size_t, size_t> calculate_padding_requirements(const PaddingSize &padding);

    size_t           _total_size;
    int              _fixed_point_position;
    size_t           _offset_first_element_in_bytes;
    Strides          _strides_in_bytes;
    size_t           _num_channels;
    TensorShape      _tensor_shape;
    DataType         _data_type;
    Format           _format;
    bool             _is_resizable;
    ValidRegion      _valid_region;
    PaddingSize      _padding;
    QuantizationInfo _quantization_info;
};
}
#endif /*__ARM_COMPUTE_TENSORINFO_H__ */
