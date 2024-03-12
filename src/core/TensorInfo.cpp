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
#include "arm_compute/core/TensorInfo.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/Utils.h"

#include <memory>

namespace arm_compute
{
TensorInfo::TensorInfo()
    : _total_size(0),
      _offset_first_element_in_bytes(0),
      _strides_in_bytes(),
      _num_channels(0),
      _tensor_shape(),
      _dims_state(),
      _data_type(DataType::UNKNOWN),
      _format(Format::UNKNOWN),
      _text_format(TextFormat::UTF8),
      _is_resizable{true},
      _valid_region{Coordinates(), _tensor_shape},
      _padding{0},
      _quantization_info(),
      _data_layout(DataLayout::NCHW),
      _are_values_constant(true),
      _id(invalid_tensor_id),
      _lock_paddings(false)
{
}

TensorInfo::TensorInfo(const ITensorInfo &info) : TensorInfo()
{
    _total_size                    = info.total_size();
    _offset_first_element_in_bytes = info.offset_first_element_in_bytes();
    _strides_in_bytes              = info.strides_in_bytes();
    _num_channels                  = info.num_channels();
    _tensor_shape                  = info.tensor_shape();
    _dims_state                    = info.tensor_dims_state();
    _data_type                     = info.data_type();
    _format                        = info.format();
    _text_format                   = info.text_format();
    _is_resizable                  = info.is_resizable();
    _valid_region                  = info.valid_region();
    _padding                       = info.padding();
    _quantization_info             = info.quantization_info();
    _data_layout                   = info.data_layout();
    _are_values_constant           = info.are_values_constant();
    _id                            = info.id();
    _lock_paddings                 = info.lock_paddings();
}

TensorInfo::TensorInfo(const TensorInfo &info) : TensorInfo()
{
    _total_size                    = info.total_size();
    _offset_first_element_in_bytes = info.offset_first_element_in_bytes();
    _strides_in_bytes              = info.strides_in_bytes();
    _num_channels                  = info.num_channels();
    _tensor_shape                  = info.tensor_shape();
    _dims_state                    = info.tensor_dims_state();
    _data_type                     = info.data_type();
    _format                        = info.format();
    _text_format                   = info.text_format();
    _is_resizable                  = info.is_resizable();
    _valid_region                  = info.valid_region();
    _padding                       = info.padding();
    _quantization_info             = info.quantization_info();
    _data_layout                   = info.data_layout();
    _are_values_constant           = info.are_values_constant();
    _id                            = info.id();
    _lock_paddings                 = false;
}
TensorInfo::TensorInfo(Format format) : TensorInfo(TensorShape(), format)
{
}

/* transformer */
TensorInfo::TensorInfo(unsigned int length, TextFormat format): TensorInfo(TensorShape(length), format)
{
}

TensorInfo::TensorInfo(unsigned int width, unsigned int height, Format format)
    : TensorInfo(TensorShape(width, height), format)
{
}

TensorInfo::TensorInfo(const TensorShape &tensor_shape, Format format) : TensorInfo()
{
    init(tensor_shape, format);
}

TensorInfo::TensorInfo(const TensorShape &tensor_shape, TextFormat format) : TensorInfo()
{
    init(tensor_shape, format);
}

TensorInfo::TensorInfo(size_t num_channels, DataType data_type) : TensorInfo()
{
    init(TensorShape(), num_channels, data_type);
}

TensorInfo::TensorInfo(const TensorShape &tensor_shape, size_t num_channels, DataType data_type) : TensorInfo()
{
    init(tensor_shape, num_channels, data_type);
}

TensorInfo::TensorInfo(const TensorShape &tensor_shape,
                       size_t             num_channels,
                       DataType           data_type,
                       QuantizationInfo   quantization_info)
    : TensorInfo()
{
    init(tensor_shape, num_channels, data_type);
    _quantization_info = std::move(quantization_info);
}

TensorInfo::TensorInfo(const TensorShape &tensor_shape, size_t num_channels, DataType data_type, DataLayout data_layout)
    : TensorInfo()
{
    init(tensor_shape, num_channels, data_type);
    _data_layout = data_layout;
}

void TensorInfo::init(Format format)
{
    init(TensorShape(), format);
}

void TensorInfo::init(const TensorShape &tensor_shape, Format format)
{
    size_t         num_channels = num_channels_from_format(format);
    const DataType type         = data_type_from_format(format);

    init(tensor_shape, num_channels, type);

    _format = format;
}

void TensorInfo::init(const TensorShape &tensor_shape, TextFormat format)
{
    const DataType type         = data_type_from_format(format);

    init(tensor_shape, type);

    _text_format = format;
}

void TensorInfo::init(const TensorShape &tensor_shape,
                      Format             format,
                      const Strides     &strides_in_bytes,
                      size_t             offset_first_element_in_bytes,
                      size_t             total_size_in_bytes)
{
    size_t         num_channels = num_channels_from_format(format);
    const DataType type         = data_type_from_format(format);

    init(tensor_shape, num_channels, type, strides_in_bytes, offset_first_element_in_bytes, total_size_in_bytes);

    _format = format;
}

void TensorInfo::init(size_t num_channels, DataType data_type)
{
    init(TensorShape(), num_channels, data_type);
}

void TensorInfo::init(const TensorShape &tensor_shape, size_t num_channels, DataType data_type)
{
    ARM_COMPUTE_ERROR_ON(num_channels == 0);

    _data_type    = data_type;
    _num_channels = num_channels;
    _format       = Format::UNKNOWN;

    set_tensor_shape(tensor_shape);
}

void TensorInfo::init(const TensorShape &tensor_shape, DataType data_type)
{
    ARM_COMPUTE_ERROR_ON(num_channels == 0);

    _data_type    = data_type;
    _format       = Format::UNKNOWN;

    set_tensor_shape(tensor_shape);
}

void TensorInfo::init(const TensorShape &tensor_shape,
                      size_t             num_channels,
                      DataType           data_type,
                      const Strides     &strides_in_bytes,
                      size_t             offset_first_element_in_bytes,
                      size_t             total_size_in_bytes)
{
    ARM_COMPUTE_ERROR_ON(num_channels == 0);

    _data_type                     = data_type;
    _num_channels                  = num_channels;
    _format                        = Format::UNKNOWN;
    _tensor_shape                  = tensor_shape;
    _offset_first_element_in_bytes = offset_first_element_in_bytes;
    _strides_in_bytes              = strides_in_bytes;
    _total_size                    = total_size_in_bytes;

    _valid_region = ValidRegion{Coordinates(), _tensor_shape};
}

size_t TensorInfo::init_auto_padding(const TensorShape &tensor_shape, Format format)
{
    const size_t   num_channels = num_channels_from_format(format);
    const DataType type         = data_type_from_format(format);
    size_t         total_size   = init_auto_padding(tensor_shape, num_channels, type);

    _format = format;

    return total_size;
}

size_t TensorInfo::init_auto_padding(const TensorShape &tensor_shape, size_t num_channels, DataType data_type)
{
    ARM_COMPUTE_ERROR_ON(num_channels == 0);

    _data_type    = data_type;
    _num_channels = num_channels;
    _format       = Format::UNKNOWN;
    _tensor_shape = tensor_shape;

    _valid_region = ValidRegion{Coordinates(), _tensor_shape};

    auto_padding();

    return _total_size;
}

bool TensorInfo::auto_padding()
{
    ARM_COMPUTE_ERROR_ON(!_is_resizable);

    // Some kernels compute 32 elements at the time, worst case scenario they
    // will read 32 values after the last element
    const size_t extra_pad_x = _tensor_shape.num_dimensions() < 1 ? 0 : 32;
    const size_t pad_x       = _tensor_shape.num_dimensions() < 1 ? 0 : 4;
    const size_t pad_y       = _tensor_shape.num_dimensions() < 2 ? 0 : 4;

    return extend_padding(PaddingSize(pad_y, pad_x + extra_pad_x, pad_y, pad_x));
}

std::tuple<Strides, size_t, size_t> TensorInfo::calculate_padding_requirements(const PaddingSize &padding)
{
    // Calculate resulting stride for the X, Y and Z dimension
    const size_t stride_x = element_size();
    const size_t stride_y = (padding.left + _tensor_shape[0] + padding.right) * stride_x;
    const size_t stride_z = (padding.top + _tensor_shape[1] + padding.bottom) * stride_y;

    Strides      required_strides;
    size_t       required_total_size           = 0;
    const size_t required_offset_first_element = padding.left * stride_x + padding.top * stride_y;

    switch (_tensor_shape.num_dimensions())
    {
        case 0:
        {
            if (_tensor_shape.total_size() > 0)
            {
                required_strides    = Strides(stride_x, stride_x);
                required_total_size = stride_z;
            }
            break;
        }
        case 1:
            required_strides    = compute_strides(*this, stride_x, stride_y);
            required_total_size = stride_z;
            break;
        case 2:
            required_strides    = compute_strides(*this, stride_x, stride_y);
            required_total_size = stride_z;
            break;
        default:
        {
            required_strides = compute_strides(*this, stride_x, stride_y, stride_z);

            const unsigned int idx_last_dimension = _tensor_shape.num_dimensions() - 1;

            required_total_size =
                static_cast<size_t>(_tensor_shape[idx_last_dimension]) * required_strides[idx_last_dimension];
            break;
        }
    }

    return std::make_tuple(required_strides, required_offset_first_element, required_total_size);
}

ITensorInfo &TensorInfo::set_lock_paddings(bool flag)
{
    _lock_paddings = flag;
    return *this;
}

bool TensorInfo::lock_paddings() const
{
    return _lock_paddings;
}

bool TensorInfo::extend_padding(const PaddingSize &padding)
{
    ARM_COMPUTE_ERROR_ON(_lock_paddings);
    ARM_COMPUTE_ERROR_ON(!_is_resizable);

    bool updated = false;

    if (padding.top > _padding.top)
    {
        _padding.top = padding.top;
        updated      = true;
    }

    if (padding.right > _padding.right)
    {
        _padding.right = padding.right;
        updated        = true;
    }

    if (padding.bottom > _padding.bottom)
    {
        _padding.bottom = padding.bottom;
        updated         = true;
    }

    if (padding.left > _padding.left)
    {
        _padding.left = padding.left;
        updated       = true;
    }

    std::tie(_strides_in_bytes, _offset_first_element_in_bytes, _total_size) = calculate_padding_requirements(_padding);

    return updated;
}

std::unique_ptr<ITensorInfo> TensorInfo::clone() const
{
    return std::make_unique<TensorInfo>(*this);
}

ITensorInfo &TensorInfo::set_data_type(DataType data_type)
{
    _data_type = data_type;
    _format    = Format::UNKNOWN;
    return set_tensor_shape(tensor_shape()); // Force total size and strides to update
}

ITensorInfo &TensorInfo::set_num_channels(int num_channels)
{
    _num_channels = num_channels;
    _format       = Format::UNKNOWN;
    return *this;
}

ITensorInfo &TensorInfo::set_format(Format format)
{
    _format = format;

    if (_data_type == DataType::UNKNOWN)
    {
        _num_channels = num_channels_from_format(format);
        _data_type    = data_type_from_format(format);
    }
    else
    {
        ARM_COMPUTE_ERROR_ON(num_channels_from_format(format) != _num_channels);
        ARM_COMPUTE_ERROR_ON(data_type_from_format(format) != _data_type);
    }
    return *this;
}


ITensorInfo &TensorInfo::set_text_format(TextFormat text_format)
{
    _text_format = text_format;

    return *this;
}

ITensorInfo &TensorInfo::set_tensor_shape(const TensorShape &shape)
{
    _tensor_shape                  = shape;
    _offset_first_element_in_bytes = 0;
    _strides_in_bytes              = compute_strides(*this);

    if (_tensor_shape.num_dimensions() == 0)
    {
        _total_size = _strides_in_bytes[0];
    }
    else
    {
        const unsigned int idx_last_dimension = _tensor_shape.num_dimensions() - 1;
        _total_size = static_cast<size_t>(_tensor_shape[idx_last_dimension]) * _strides_in_bytes[idx_last_dimension];
    }

    std::tie(_strides_in_bytes, _offset_first_element_in_bytes, _total_size) = calculate_padding_requirements(_padding);

    _valid_region = ValidRegion{Coordinates(), _tensor_shape};
    return *this;
}

ITensorInfo &TensorInfo::set_tensor_dims_state(const TensorDimsState &state)
{
    _dims_state = state;
    return *this;
}

ITensorInfo &TensorInfo::set_quantization_info(const QuantizationInfo &quantization_info)
{
    _quantization_info = quantization_info;
    return *this;
}

ITensorInfo &TensorInfo::set_data_layout(const DataLayout &data_layout)
{
    _data_layout = data_layout;
    return *this;
}

ITensorInfo &TensorInfo::reset_padding()
{
    _padding = PaddingSize();
    if (((_format != Format::UNKNOWN) || (_data_type != DataType::UNKNOWN)) && _total_size != 0)
    {
        std::tie(_strides_in_bytes, _offset_first_element_in_bytes, _total_size) =
            calculate_padding_requirements(_padding);
    }
    return *this;
}

int32_t TensorInfo::offset_element_in_bytes(const Coordinates &pos) const
{
    ARM_COMPUTE_ERROR_ON_COORDINATES_DIMENSIONS_GTE(pos, _tensor_shape.num_dimensions());

    int32_t offset = _offset_first_element_in_bytes;

    for (size_t i = 0; i < _tensor_shape.num_dimensions(); ++i)
    {
        offset += pos[i] * _strides_in_bytes[i];
    }

    return offset;
}
} // namespace arm_compute
