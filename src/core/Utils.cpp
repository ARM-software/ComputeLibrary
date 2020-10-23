/*
 * Copyright (c) 2016-2020 Arm Limited.
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

#include "arm_compute/core/Helpers.h"

#include "arm_compute/core/Utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>

namespace arm_compute
{
std::string read_file(const std::string &filename, bool binary)
{
    std::string   out;
    std::ifstream fs;

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        fs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        std::ios_base::openmode mode = std::ios::in;

        if(binary)
        {
            mode |= std::ios::binary;
        }

        fs.open(filename, mode);

        // Go to the end of the file
        fs.seekg(0, std::ios::end);
        // Reserve the memory required to store the file's content
        out.reserve(fs.tellg());
        // Go back to the beginning of the file
        fs.seekg(0, std::ios::beg);
        // Copy the content of the file
        out.assign(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::ifstream::failure &e)
    {
        ARM_COMPUTE_ERROR_VAR("Accessing %s: %s", filename.c_str(), e.what());
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */

    return out;
}

const std::string &string_from_format(Format format)
{
    static std::map<Format, const std::string> formats_map =
    {
        { Format::UNKNOWN, "UNKNOWN" },
        { Format::U8, "U8" },
        { Format::S16, "S16" },
        { Format::U16, "U16" },
        { Format::S32, "S32" },
        { Format::U32, "U32" },
        { Format::F16, "F16" },
        { Format::F32, "F32" },
        { Format::UV88, "UV88" },
        { Format::RGB888, "RGB888" },
        { Format::RGBA8888, "RGBA8888" },
        { Format::YUV444, "YUV444" },
        { Format::YUYV422, "YUYV422" },
        { Format::NV12, "NV12" },
        { Format::NV21, "NV21" },
        { Format::IYUV, "IYUV" },
        { Format::UYVY422, "UYVY422" }
    };

    return formats_map[format];
}

const std::string &string_from_channel(Channel channel)
{
    static std::map<Channel, const std::string> channels_map =
    {
        { Channel::UNKNOWN, "UNKNOWN" },
        { Channel::R, "R" },
        { Channel::G, "G" },
        { Channel::B, "B" },
        { Channel::A, "A" },
        { Channel::Y, "Y" },
        { Channel::U, "U" },
        { Channel::V, "V" },
        { Channel::C0, "C0" },
        { Channel::C1, "C1" },
        { Channel::C2, "C2" },
        { Channel::C3, "C3" }
    };

    return channels_map[channel];
}

const std::string &string_from_data_layout(DataLayout dl)
{
    static std::map<DataLayout, const std::string> dl_map =
    {
        { DataLayout::UNKNOWN, "UNKNOWN" },
        { DataLayout::NCHW, "NCHW" },
        { DataLayout::NHWC, "NHWC" },
    };

    return dl_map[dl];
}

const std::string &string_from_data_type(DataType dt)
{
    static std::map<DataType, const std::string> dt_map =
    {
        { DataType::UNKNOWN, "UNKNOWN" },
        { DataType::S8, "S8" },
        { DataType::U8, "U8" },
        { DataType::S16, "S16" },
        { DataType::U16, "U16" },
        { DataType::S32, "S32" },
        { DataType::U32, "U32" },
        { DataType::S64, "S64" },
        { DataType::U64, "U64" },
        { DataType::F16, "F16" },
        { DataType::F32, "F32" },
        { DataType::F64, "F64" },
        { DataType::SIZET, "SIZET" },
        { DataType::QSYMM8, "QSYMM8" },
        { DataType::QSYMM8_PER_CHANNEL, "QSYMM8_PER_CHANNEL" },
        { DataType::QASYMM8, "QASYMM8" },
        { DataType::QASYMM8_SIGNED, "QASYMM8_SIGNED" },
        { DataType::QSYMM16, "QSYMM16" },
        { DataType::QASYMM16, "QASYMM16" },
    };

    return dt_map[dt];
}

const std::string &string_from_activation_func(ActivationLayerInfo::ActivationFunction act)
{
    static std::map<ActivationLayerInfo::ActivationFunction, const std::string> act_map =
    {
        { ActivationLayerInfo::ActivationFunction::ABS, "ABS" },
        { ActivationLayerInfo::ActivationFunction::LINEAR, "LINEAR" },
        { ActivationLayerInfo::ActivationFunction::LOGISTIC, "LOGISTIC" },
        { ActivationLayerInfo::ActivationFunction::RELU, "RELU" },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, "BRELU" },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, "LU_BRELU" },
        { ActivationLayerInfo::ActivationFunction::LEAKY_RELU, "LRELU" },
        { ActivationLayerInfo::ActivationFunction::SOFT_RELU, "SRELU" },
        { ActivationLayerInfo::ActivationFunction::ELU, "ELU" },
        { ActivationLayerInfo::ActivationFunction::SQRT, "SQRT" },
        { ActivationLayerInfo::ActivationFunction::SQUARE, "SQUARE" },
        { ActivationLayerInfo::ActivationFunction::TANH, "TANH" },
        { ActivationLayerInfo::ActivationFunction::IDENTITY, "IDENTITY" },
        { ActivationLayerInfo::ActivationFunction::HARD_SWISH, "HARD_SWISH" }

    };

    return act_map[act];
}

const std::string &string_from_matrix_pattern(MatrixPattern pattern)
{
    static std::map<MatrixPattern, const std::string> pattern_map =
    {
        { MatrixPattern::BOX, "BOX" },
        { MatrixPattern::CROSS, "CROSS" },
        { MatrixPattern::DISK, "DISK" },
        { MatrixPattern::OTHER, "OTHER" },
    };

    return pattern_map[pattern];
}

const std::string &string_from_non_linear_filter_function(NonLinearFilterFunction function)
{
    static std::map<NonLinearFilterFunction, const std::string> func_map =
    {
        { NonLinearFilterFunction::MAX, "MAX" },
        { NonLinearFilterFunction::MEDIAN, "MEDIAN" },
        { NonLinearFilterFunction::MIN, "MIN" },
    };

    return func_map[function];
}

const std::string &string_from_interpolation_policy(InterpolationPolicy policy)
{
    static std::map<InterpolationPolicy, const std::string> interpolation_policy_map =
    {
        { InterpolationPolicy::AREA, "AREA" },
        { InterpolationPolicy::BILINEAR, "BILINEAR" },
        { InterpolationPolicy::NEAREST_NEIGHBOR, "NEAREST_NEIGHBOUR" },
    };

    return interpolation_policy_map[policy];
}

const std::string &string_from_border_mode(BorderMode border_mode)
{
    static std::map<BorderMode, const std::string> border_mode_map =
    {
        { BorderMode::UNDEFINED, "UNDEFINED" },
        { BorderMode::CONSTANT, "CONSTANT" },
        { BorderMode::REPLICATE, "REPLICATE" },
    };

    return border_mode_map[border_mode];
}

const std::string &string_from_norm_type(NormType type)
{
    static std::map<NormType, const std::string> norm_type_map =
    {
        { NormType::IN_MAP_1D, "IN_MAP_1D" },
        { NormType::IN_MAP_2D, "IN_MAP_2D" },
        { NormType::CROSS_MAP, "CROSS_MAP" },
    };

    return norm_type_map[type];
}

const std::string &string_from_pooling_type(PoolingType type)
{
    static std::map<PoolingType, const std::string> pool_type_map =
    {
        { PoolingType::MAX, "MAX" },
        { PoolingType::AVG, "AVG" },
        { PoolingType::L2, "L2" },
    };

    return pool_type_map[type];
}

const std::string &string_from_gemmlowp_output_stage(GEMMLowpOutputStageType output_stage)
{
    static std::map<GEMMLowpOutputStageType, const std::string> output_stage_map =
    {
        { GEMMLowpOutputStageType::NONE, "" },
        { GEMMLowpOutputStageType::QUANTIZE_DOWN, "quantize_down" },
        { GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, "quantize_down_fixedpoint" },
        { GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT, "quantize_down_float" }
    };

    return output_stage_map[output_stage];
}

std::string string_from_pixel_value(const PixelValue &value, const DataType data_type)
{
    std::stringstream ss;
    std::string       converted_string;

    switch(data_type)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            // Needs conversion to 32 bit, otherwise interpreted as ASCII values
            ss << uint32_t(value.get<uint8_t>());
            converted_string = ss.str();
            break;
        case DataType::S8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            // Needs conversion to 32 bit, otherwise interpreted as ASCII values
            ss << int32_t(value.get<int8_t>());
            converted_string = ss.str();
            break;
        case DataType::U16:
        case DataType::QASYMM16:
            ss << value.get<uint16_t>();
            converted_string = ss.str();
            break;
        case DataType::S16:
        case DataType::QSYMM16:
            ss << value.get<int16_t>();
            converted_string = ss.str();
            break;
        case DataType::U32:
            ss << value.get<uint32_t>();
            converted_string = ss.str();
            break;
        case DataType::S32:
            ss << value.get<int32_t>();
            converted_string = ss.str();
            break;
        case DataType::F32:
            converted_string = float_to_string_with_full_precision(value.get<float>());
            break;
        case DataType::F16:
            static_assert(sizeof(half) == 2, "Half must be 16 bit");
            ss << value.get<half>();
            converted_string = ss.str();
            break;
        default:
            ARM_COMPUTE_ERROR("Not handled");
    }

    return converted_string;
}

DataType data_type_from_name(const std::string &name)
{
    static const std::map<std::string, DataType> data_types =
    {
        { "f16", DataType::F16 },
        { "f32", DataType::F32 },
        { "qasymm8", DataType::QASYMM8 },
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return data_types.at(utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        ARM_COMPUTE_ERROR_VAR("Invalid data type name: %s", name.c_str());
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}

std::string lower_string(const std::string &val)
{
    std::string res = val;
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

PadStrideInfo calculate_same_pad(TensorShape input_shape, TensorShape weights_shape, PadStrideInfo conv_info, DataLayout data_layout, const Size2D &dilation,
                                 const DimensionRoundingType &rounding_type)
{
    const auto &strides = conv_info.stride();
    ARM_COMPUTE_ERROR_ON_MSG((strides.first < 1 || strides.second < 1), "Stride values should be greater than or equal to 1.");

    const unsigned int width_idx     = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int in_width      = input_shape[width_idx];
    const unsigned int in_height     = input_shape[height_idx];
    const unsigned int kernel_width  = weights_shape[width_idx];
    const unsigned int kernel_height = weights_shape[height_idx];

    // Calculate output dimensions
    const auto         is_ceil    = static_cast<unsigned int>(rounding_type == DimensionRoundingType::CEIL);
    const unsigned int out_width  = ((in_width - is_ceil) + strides.first - 1) / strides.first + is_ceil;
    const unsigned int out_height = ((in_height - is_ceil) + strides.second - 1) / strides.second + is_ceil;

    // Calculate effective weights sizes
    const int real_weight_width  = (kernel_width - 1) * dilation.x() + 1;
    const int real_weight_height = (kernel_height - 1) * dilation.y() + 1;

    // Calculate total pad
    const int pad_width  = std::max(0, static_cast<int>((out_width - 1) * strides.first + real_weight_width - in_width));
    const int pad_height = std::max(0, static_cast<int>((out_height - 1) * strides.second + real_weight_height - in_height));

    // Calculate individual paddings
    const unsigned int pad_left   = pad_width / 2;
    const unsigned int pad_top    = pad_height / 2;
    const unsigned int pad_right  = pad_width - pad_left;
    const unsigned int pad_bottom = pad_height - pad_top;

    PadStrideInfo same_info(strides.first, strides.second, pad_left, pad_right, pad_top, pad_bottom, rounding_type);

    // Check for correctness of predicted output shape against the one calculated using the generated info
    const auto out_dims = scaled_dimensions(in_width, in_height, kernel_width, kernel_height, same_info, dilation);
    ARM_COMPUTE_ERROR_ON(out_dims.first != out_width || out_dims.second != out_height);
    ARM_COMPUTE_UNUSED(out_dims);

    return same_info;
}

std::pair<unsigned int, unsigned int> deconvolution_output_dimensions(unsigned int in_width, unsigned int in_height,
                                                                      unsigned int kernel_width, unsigned int kernel_height,
                                                                      const PadStrideInfo &pad_stride_info)
{
    const unsigned int pad_left   = pad_stride_info.pad_left();
    const unsigned int pad_top    = pad_stride_info.pad_top();
    const unsigned int pad_right  = pad_stride_info.pad_right();
    const unsigned int pad_bottom = pad_stride_info.pad_bottom();
    const unsigned int stride_x   = pad_stride_info.stride().first;
    const unsigned int stride_y   = pad_stride_info.stride().second;

    ARM_COMPUTE_ERROR_ON(in_width < 1 || in_height < 1);
    ARM_COMPUTE_ERROR_ON(((in_width - 1) * stride_x + kernel_width) < (pad_left + pad_right));
    ARM_COMPUTE_ERROR_ON(((in_height - 1) * stride_y + kernel_height) < (pad_top + pad_bottom));
    const int w = stride_x * (in_width - 1) + kernel_width - (pad_left + pad_right);
    const int h = stride_y * (in_height - 1) + kernel_height - (pad_top + pad_bottom);

    return std::make_pair<unsigned int, unsigned int>(w, h);
}

std::pair<unsigned int, unsigned int> scaled_dimensions(int width, int height,
                                                        int kernel_width, int kernel_height,
                                                        const PadStrideInfo &pad_stride_info,
                                                        const Size2D        &dilation)
{
    const int dilation_x = dilation.x();
    const int dilation_y = dilation.y();
    const int pad_left   = pad_stride_info.pad_left();
    const int pad_top    = pad_stride_info.pad_top();
    const int pad_right  = pad_stride_info.pad_right();
    const int pad_bottom = pad_stride_info.pad_bottom();
    const int stride_x   = pad_stride_info.stride().first;
    const int stride_y   = pad_stride_info.stride().second;
    int       w          = 0;
    int       h          = 0;
    switch(pad_stride_info.round())
    {
        case DimensionRoundingType::FLOOR:
            w = static_cast<int>(std::floor((static_cast<float>(width + pad_left + pad_right - (dilation_x * (kernel_width - 1) + 1)) / stride_x) + 1));
            h = static_cast<int>(std::floor((static_cast<float>(height + pad_top + pad_bottom - (dilation_y * (kernel_height - 1) + 1)) / stride_y) + 1));
            break;
        case DimensionRoundingType::CEIL:
            w = static_cast<int>(std::ceil((static_cast<float>(width + pad_left + pad_right - (dilation_x * (kernel_width - 1) + 1)) / stride_x) + 1));
            h = static_cast<int>(std::ceil((static_cast<float>(height + pad_top + pad_bottom - (dilation_y * (kernel_height - 1) + 1)) / stride_y) + 1));
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported rounding type");
    }

    w = std::max(1, w);
    h = std::max(1, h);
    return std::make_pair(static_cast<unsigned int>(w), static_cast<unsigned int>(h));
}

bool needs_serialized_reduction(ReductionOperation op, DataType dt, unsigned int axis)
{
    const bool is_min_max        = (op == ReductionOperation::MAX || op == ReductionOperation::MIN);
    const bool is_quantized_type = is_data_type_quantized(dt);
    const bool is_first_dim      = (axis == 0);

    return !is_first_dim || is_min_max || is_quantized_type;
}

QuantizationInfo get_softmax_output_quantization_info(DataType input_type, bool is_log)
{
    // Note: Output quantization info for softmax should always have
    // * Softmax with QASYMM8: scale = 1/256, offset = 0
    // * Softmax with QASYMM8_SIGNED: scale = 1/256, offset = -128
    // * LogSoftmax with QASYMM8: scale = 1/256, offset = 0
    // * LogSoftmax with QASYMM8_SIGNED: scale = 16/256, offset = 127
    if(is_data_type_quantized_asymmetric_signed(input_type))
    {
        if(is_log)
        {
            return QuantizationInfo(16.f / 256, 127);
        }
        else
        {
            return QuantizationInfo(1.f / 256, -128);
        }
    }
    return QuantizationInfo(1.f / 256, 0);
}

std::pair<int32_t, int32_t> get_quantized_activation_min_max(ActivationLayerInfo act_info, DataType data_type, UniformQuantizationInfo oq_info)
{
    const bool is_qasymm8_signed = is_data_type_quantized_asymmetric_signed(data_type);
    const auto a                 = act_info.a();
    const auto b                 = act_info.b();
    const int  a_int             = is_qasymm8_signed ? quantize_qasymm8_signed(a, oq_info) : quantize_qasymm8(a, oq_info);
    const int  b_int             = is_qasymm8_signed ? quantize_qasymm8_signed(b, oq_info) : quantize_qasymm8(b, oq_info);
    const auto type_max_value    = std::get<1>(get_min_max(data_type)).get<int32_t>();

    const int32_t min_activation = act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU ? oq_info.offset : b_int;
    const int32_t max_activation = act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU ? type_max_value : a_int;

    return std::make_pair(min_activation, max_activation);
}

std::unordered_map<const ITensorInfo *, PaddingSize> get_padding_info(std::initializer_list<const ITensor *> tensors)
{
    std::unordered_map<const ITensorInfo *, PaddingSize> res;

    for(const ITensor *tensor : tensors)
    {
        if(tensor)
        {
            res.insert({ tensor->info(), tensor->info()->padding() });
        }
    }

    return res;
}

std::unordered_map<const ITensorInfo *, PaddingSize> get_padding_info(std::initializer_list<const ITensorInfo *> infos)
{
    std::unordered_map<const ITensorInfo *, PaddingSize> res;

    for(const ITensorInfo *info : infos)
    {
        if(info)
        {
            res.insert({ info, info->padding() });
        }
    }

    return res;
}

bool has_padding_changed(const std::unordered_map<const ITensorInfo *, PaddingSize> &padding_map)
{
    return std::find_if(padding_map.begin(), padding_map.end(), [](const std::pair<const ITensorInfo *, PaddingSize> &padding_info)
    {
        return (padding_info.first->padding() != padding_info.second);
    })
    != padding_map.end();
}

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
void print_consecutive_elements(std::ostream &s, DataType dt, const uint8_t *ptr, unsigned int n, int stream_width, const std::string &element_delim)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            print_consecutive_elements_impl<uint8_t>(s, ptr, n, stream_width, element_delim);
            break;
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            print_consecutive_elements_impl<int8_t>(s, reinterpret_cast<const int8_t *>(ptr), n, stream_width, element_delim);
            break;
        case DataType::U16:
        case DataType::QASYMM16:
            print_consecutive_elements_impl<uint16_t>(s, reinterpret_cast<const uint16_t *>(ptr), n, stream_width, element_delim);
            break;
        case DataType::S16:
        case DataType::QSYMM16:
            print_consecutive_elements_impl<int16_t>(s, reinterpret_cast<const int16_t *>(ptr), n, stream_width, element_delim);
            break;
        case DataType::U32:
            print_consecutive_elements_impl<uint32_t>(s, reinterpret_cast<const uint32_t *>(ptr), n, stream_width, element_delim);
            break;
        case DataType::S32:
            print_consecutive_elements_impl<int32_t>(s, reinterpret_cast<const int32_t *>(ptr), n, stream_width, element_delim);
            break;
        case DataType::BFLOAT16:
            print_consecutive_elements_impl<bfloat16>(s, reinterpret_cast<const bfloat16 *>(ptr), n, stream_width, element_delim);
            break;
        case DataType::F16:
            print_consecutive_elements_impl<half>(s, reinterpret_cast<const half *>(ptr), n, stream_width, element_delim);
            break;
        case DataType::F32:
            print_consecutive_elements_impl<float>(s, reinterpret_cast<const float *>(ptr), n, stream_width, element_delim);
            break;
        default:
            ARM_COMPUTE_ERROR("Undefined element size for given data type");
    }
}

int max_consecutive_elements_display_width(std::ostream &s, DataType dt, const uint8_t *ptr, unsigned int n)
{
    switch(dt)
    {
        case DataType::U8:
        case DataType::QASYMM8:
            return max_consecutive_elements_display_width_impl<uint8_t>(s, ptr, n);
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            return max_consecutive_elements_display_width_impl<int8_t>(s, reinterpret_cast<const int8_t *>(ptr), n);
        case DataType::U16:
        case DataType::QASYMM16:
            return max_consecutive_elements_display_width_impl<uint16_t>(s, reinterpret_cast<const uint16_t *>(ptr), n);
        case DataType::S16:
        case DataType::QSYMM16:
            return max_consecutive_elements_display_width_impl<int16_t>(s, reinterpret_cast<const int16_t *>(ptr), n);
        case DataType::U32:
            return max_consecutive_elements_display_width_impl<uint32_t>(s, reinterpret_cast<const uint32_t *>(ptr), n);
        case DataType::S32:
            return max_consecutive_elements_display_width_impl<int32_t>(s, reinterpret_cast<const int32_t *>(ptr), n);
        case DataType::BFLOAT16:
            return max_consecutive_elements_display_width_impl<bfloat16>(s, reinterpret_cast<const bfloat16 *>(ptr), n);
        case DataType::F16:
            return max_consecutive_elements_display_width_impl<half>(s, reinterpret_cast<const half *>(ptr), n);
        case DataType::F32:
            return max_consecutive_elements_display_width_impl<float>(s, reinterpret_cast<const float *>(ptr), n);
        default:
            ARM_COMPUTE_ERROR("Undefined element size for given data type");
    }
    return 0;
}
#endif /* ARM_COMPUTE_ASSERTS_ENABLED */

} // namespace arm_compute