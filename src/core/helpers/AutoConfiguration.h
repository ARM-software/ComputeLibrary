/*
* Copyright (c) 2020, 2023 Arm Limited.
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
#ifndef SRC_CORE_HELPERS_AUTOCONFIGURATION_H
#define SRC_CORE_HELPERS_AUTOCONFIGURATION_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** Auto initialize the tensor info (shape, number of channels and data type) if the current assignment is empty.
 *
 * @param[in,out] info              Tensor info used to check and assign.
 * @param[in]     shape             New shape.
 * @param[in]     num_channels      New number of channels.
 * @param[in]     data_type         New data type
 * @param[in]     quantization_info (Optional) New quantization info
 *
 * @return True if the tensor info has been initialized
 */
inline bool auto_init_if_empty(ITensorInfo       &info,
                               const TensorShape &shape,
                               int num_channels, DataType data_type,
                               QuantizationInfo quantization_info = QuantizationInfo())
{
    if(info.tensor_shape().total_size() == 0)
    {
        info.set_data_type(data_type);
        info.set_num_channels(num_channels);
        info.set_tensor_shape(shape);
        info.set_quantization_info(quantization_info);
        return true;
    }

    return false;
}

/** Auto initialize the tensor info using another tensor info.
 *
 * (COMPMID-6012) This method should remain in sync with the fields of ITensorInfo that have setters.
 *
 *
 * @param info_sink   Tensor info used to check and assign
 * @param info_source Tensor info used to assign
 *
 *
 * @return True if the tensor info has been initialized
 */
inline bool auto_init_if_empty(ITensorInfo &info_sink, const ITensorInfo &info_source)
{
    if(info_sink.tensor_shape().total_size() == 0)
    {
        info_sink.set_data_type(info_source.data_type());
        info_sink.set_num_channels(info_source.num_channels());
        info_sink.set_tensor_shape(info_source.tensor_shape());
        info_sink.set_quantization_info(info_source.quantization_info());
        info_sink.set_data_layout(info_source.data_layout());
        info_sink.set_are_values_constant(info_source.are_values_constant());
        return true;
    }

    return false;
}

/** Set the shape to the specified value if the current assignment is empty.
 *
 * @param[in,out] info  Tensor info used to check and assign.
 * @param[in]     shape New shape.
 *
 * @return True if the shape has been changed.
 */
inline bool set_shape_if_empty(ITensorInfo &info, const TensorShape &shape)
{
    if(info.tensor_shape().total_size() == 0)
    {
        info.set_tensor_shape(shape);
        return true;
    }

    return false;
}

/** Set the format, data type and number of channels to the specified value if
 * the current data type is unknown.
 *
 * @param[in,out] info   Tensor info used to check and assign.
 * @param[in]     format New format.
 *
 * @return True if the format has been changed.
 */
inline bool set_format_if_unknown(ITensorInfo &info, Format format)
{
    if(info.data_type() == DataType::UNKNOWN)
    {
        info.set_format(format);
        return true;
    }

    return false;
}

/** Set the data type and number of channels to the specified value if
 * the current data type is unknown.
 *
 * @param[in,out] info      Tensor info used to check and assign.
 * @param[in]     data_type New data type.
 *
 * @return True if the data type has been changed.
 */
inline bool set_data_type_if_unknown(ITensorInfo &info, DataType data_type)
{
    if(info.data_type() == DataType::UNKNOWN)
    {
        info.set_data_type(data_type);
        return true;
    }

    return false;
}

/** Set the data layout to the specified value if
 * the current data layout is unknown.
 *
 * @param[in,out] info        Tensor info used to check and assign.
 * @param[in]     data_layout New data layout.
 *
 * @return True if the data type has been changed.
 */
inline bool set_data_layout_if_unknown(ITensorInfo &info, DataLayout data_layout)
{
    if(info.data_layout() == DataLayout::UNKNOWN)
    {
        info.set_data_layout(data_layout);
        return true;
    }

    return false;
}

/** Set the quantization info to the specified value if
 * the current quantization info is empty and the data type of asymmetric quantized type
 *
 * @param[in,out] info              Tensor info used to check and assign.
 * @param[in]     quantization_info Quantization info
 *
 * @return True if the quantization info has been changed.
 */
inline bool set_quantization_info_if_empty(ITensorInfo &info, QuantizationInfo quantization_info)
{
    if(info.quantization_info().empty() && (is_data_type_quantized_asymmetric(info.data_type())))
    {
        info.set_quantization_info(quantization_info);
        return true;
    }

    return false;
}
} // namespace arm_compute

#endif /* SRC_CORE_HELPERS_AUTOCONFIGURATION_H */
