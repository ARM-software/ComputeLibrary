/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TENSORINFO_H
#define ARM_COMPUTE_TENSORINFO_H

#include "arm_compute/core/ITensorInfo.h"

#include "ITensorInfo.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
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
    TensorInfo(const TensorInfo &);
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
     * @param[in] num_channels It indicates the number of channels for each tensor element
     * @param[in] data_type    Data type to use for each tensor element
     */
    TensorInfo(size_t num_channels, DataType data_type);

    /** Constructor
     *
     * @param[in] tensor_shape It specifies the size for each dimension of the tensor in number of elements.
     * @param[in] num_channels It indicates the number of channels for each tensor element
     * @param[in] data_type    Data type to use for each tensor element
     */
    TensorInfo(const TensorShape &tensor_shape, size_t num_channels, DataType data_type);

    /** Constructor
     *
     * @param[in] tensor_shape It specifies the size for each dimension of the tensor in number of elements.
     * @param[in] num_channels It indicates the number of channels for each tensor element
     * @param[in] data_type    Data type to use for each tensor element
     * @param[in] data_layout  The data layout setting for the tensor data.
     */
    TensorInfo(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, DataLayout data_layout);

    /** Constructor
     *
     * @param[in] tensor_shape      It specifies the size for each dimension of the tensor in number of elements.
     * @param[in] num_channels      It indicates the number of channels for each tensor element
     * @param[in] data_type         Data type to use for each tensor element
     * @param[in] quantization_info The quantization settings for the tensor data.
     */
    TensorInfo(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, QuantizationInfo quantization_info);

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
     * @param[in] num_channels Desired number of channels for each tensor element.
     * @param[in] data_type    Data type to use for each tensor element.
     */
    void init(size_t num_channels, DataType data_type);

    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] tensor_shape Size for each dimension of the tensor in number of elements.
     * @param[in] num_channels Desired number of channels for each tensor element.
     * @param[in] data_type    Data type to use for each tensor element.
     */
    void init(const TensorShape &tensor_shape, size_t num_channels, DataType data_type);

    /** Initialize the metadata structure with the given parameters
     *
     * @param[in] tensor_shape                  Size for each dimension of the tensor in number of elements.
     * @param[in] num_channels                  Desired number of channels for each tensor element.
     * @param[in] data_type                     Data type to use for each tensor element.
     * @param[in] strides_in_bytes              Stride in bytes for accessing each dimension of the tensor.
     * @param[in] offset_first_element_in_bytes Offset in bytes from the beginning of memory allocation to access the first element.
     * @param[in] total_size_in_bytes           Size in bytes of the memory allocation (including the offset to the first element).
     */
    void init(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, const Strides &strides_in_bytes, size_t offset_first_element_in_bytes,
              size_t total_size_in_bytes);
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
    /** Initialize the metadata structure for the given tensor shape, number of channels and
     *  data type. (Padding is automatically calculated)
     *
     * @note The padding used by this method is really conservative so that the tensor can be used for most functions.
     *
     * @param[in] tensor_shape It specifies the size for each dimension of the tensor in number of elements
     * @param[in] num_channels It indicates the number of channels for each tensor element
     * @param[in] data_type    Data type to use for each tensor element
     *
     * @return Total allocation size including padding in bytes.
     */
    size_t init_auto_padding(const TensorShape &tensor_shape, size_t num_channels, DataType data_type);

    // Inherited methods overridden:
    std::unique_ptr<ITensorInfo> clone() const override;
    ITensorInfo &set_data_type(DataType data_type) override;
    ITensorInfo &set_num_channels(int num_channels) override;
    ITensorInfo &set_format(Format format) override;
    ITensorInfo &set_tensor_shape(const TensorShape &shape) override;
    ITensorInfo &set_tensor_dims_state(const TensorDimsState &state) override;
    ITensorInfo &set_quantization_info(const QuantizationInfo &quantization_info) override;
    ITensorInfo &set_data_layout(const DataLayout &data_layout) override;
    ITensorInfo &reset_padding() override;
    bool         auto_padding() override;
    ITensorInfo &set_lock_paddings(bool flag) override;
    bool lock_paddings() const override;
    bool extend_padding(const PaddingSize &padding) override;
    size_t dimension(size_t index) const override
    {
        return _tensor_shape[index];
    }
    size_t dimension(DataLayoutDimension dimension) const override
    {
        return get_data_layout_dimension_index(_data_layout, dimension);
    }
    const Strides &strides_in_bytes() const override
    {
        return _strides_in_bytes;
    }
    size_t offset_first_element_in_bytes() const override
    {
        return _offset_first_element_in_bytes;
    }
    int32_t offset_element_in_bytes(const Coordinates &pos) const override;
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
    const TensorDimsState &tensor_dims_state() const override
    {
        return _dims_state;
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
    bool is_dynamic() const override
    {
        return std::find(std::cbegin(_dims_state), std::cend(_dims_state), get_dynamic_state_value()) != std::cend(_dims_state);
    }
    bool are_values_constant() const override
    {
        return _are_values_constant;
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
    void set_valid_region(const ValidRegion &valid_region) override
    {
        _valid_region = valid_region;
    }
    QuantizationInfo quantization_info() const override
    {
        return _quantization_info;
    }
    DataLayout data_layout() const override
    {
        return _data_layout;
    }
    ITensorInfo &set_are_values_constant(bool are_values_constant) override
    {
        _are_values_constant = are_values_constant;
        return *this;
    }
    ITensorInfo::Id id() const override
    {
        return _id;
    }
    ITensorInfo &set_id(ITensorInfo::Id id) override
    {
        _id = id;
        return *this;
    }
    inline friend bool operator==(const TensorInfo &lhs, const TensorInfo &rhs);

private:
    /** Calculates strides, offset and total size resulting from the specified padding around the XY plane.
     *
     * @param[in] padding Padding around the XY plane in elements.
     */
    std::tuple<Strides, size_t, size_t> calculate_padding_requirements(const PaddingSize &padding);

    size_t           _total_size;
    size_t           _offset_first_element_in_bytes;
    Strides          _strides_in_bytes;
    size_t           _num_channels;
    TensorShape      _tensor_shape;
    TensorDimsState  _dims_state;
    DataType         _data_type;
    Format           _format;
    bool             _is_resizable;
    ValidRegion      _valid_region;
    PaddingSize      _padding;
    QuantizationInfo _quantization_info;
    DataLayout       _data_layout;
    bool             _are_values_constant;
    ITensorInfo::Id  _id;
    bool             _lock_paddings;
};

/** Check whether two tensor info are equal.
 *
 * @param[in] lhs LHS tensor info.
 * @param[in] rhs RHS tensor info.
 *
 * @return True if the given tensor infos are the same.
 */
inline bool operator==(const TensorInfo &lhs, const TensorInfo &rhs)
{
    return (lhs._total_size == rhs._total_size) && (lhs._offset_first_element_in_bytes == rhs._offset_first_element_in_bytes) && (lhs._strides_in_bytes == rhs._strides_in_bytes)
           && (lhs._num_channels == rhs._num_channels) && (lhs._tensor_shape == rhs._tensor_shape) && (lhs._dims_state == rhs._dims_state) && (lhs._data_type == rhs._data_type) && (lhs._format == rhs._format)
           && (lhs._is_resizable == rhs._is_resizable) && (lhs._valid_region == rhs._valid_region) && (lhs._padding == rhs._padding) && (lhs._quantization_info == rhs._quantization_info)
           && (lhs._data_layout == rhs._data_layout) && (lhs._are_values_constant == rhs._are_values_constant)
           && (lhs._id == rhs._id);
}
} // namespace arm_compute
#endif /*ARM_COMPUTE_TENSORINFO_H */
