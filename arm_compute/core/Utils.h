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
#ifndef ARM_COMPUTE_UTILS_H
#define ARM_COMPUTE_UTILS_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"

#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

/* Convenience / backwards compatibility includes */
#include "arm_compute/core/utils/ActivationFunctionUtils.h"
#include "arm_compute/core/utils/DataLayoutUtils.h"
#include "arm_compute/core/utils/DataTypeUtils.h"
#include "arm_compute/core/utils/FormatUtils.h"
#include "arm_compute/core/utils/InterpolationPolicyUtils.h"
#include "arm_compute/core/utils/StringUtils.h"

namespace arm_compute
{
class ITensor;
class ITensorInfo;
class ActivationLayerInfo;

/** Load an entire file in memory
 *
 * @param[in] filename Name of the file to read.
 * @param[in] binary   Is it a binary file ?
 *
 * @return The content of the file.
 */
std::string read_file(const std::string &filename, bool binary);

/** Permutes the given dimensions according the permutation vector
 *
 * @param[in,out] dimensions Dimensions to be permuted.
 * @param[in]     perm       Vector describing the permutation.
 *
 */
template <typename T>
inline void permute_strides(Dimensions<T> &dimensions, const PermutationVector &perm)
{
    const auto old_dim = utility::make_array<Dimensions<T>::num_max_dimensions>(dimensions.begin(), dimensions.end());
    for (unsigned int i = 0; i < perm.num_dimensions(); ++i)
    {
        T dimension_val = old_dim[i];
        dimensions.set(perm[i], dimension_val);
    }
}

/** Calculate padding requirements in case of SAME padding
 *
 * @param[in] input_shape   Input shape
 * @param[in] weights_shape Weights shape
 * @param[in] conv_info     Convolution information (containing strides)
 * @param[in] data_layout   (Optional) Data layout of the input and weights tensor
 * @param[in] dilation      (Optional) Dilation factor used in the convolution.
 * @param[in] rounding_type (Optional) Dimension rounding type when down-scaling.
 *
 * @return PadStrideInfo for SAME padding
 */
PadStrideInfo calculate_same_pad(TensorShape                  input_shape,
                                 TensorShape                  weights_shape,
                                 PadStrideInfo                conv_info,
                                 DataLayout                   data_layout   = DataLayout::NCHW,
                                 const Size2D                &dilation      = Size2D(1u, 1u),
                                 const DimensionRoundingType &rounding_type = DimensionRoundingType::FLOOR);

/** Returns expected width and height of the deconvolution's output tensor.
 *
 * @param[in] in_width        Width of input tensor (Number of columns)
 * @param[in] in_height       Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
std::pair<unsigned int, unsigned int> deconvolution_output_dimensions(unsigned int         in_width,
                                                                      unsigned int         in_height,
                                                                      unsigned int         kernel_width,
                                                                      unsigned int         kernel_height,
                                                                      const PadStrideInfo &pad_stride_info);

/** Returns expected width and height of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width           Width of input tensor (Number of columns)
 * @param[in] height          Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 * @param[in] dilation        (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
 *
 * @return A pair with the new width in the first position and the new height in the second.
 */
std::pair<unsigned int, unsigned int> scaled_dimensions(int                  width,
                                                        int                  height,
                                                        int                  kernel_width,
                                                        int                  kernel_height,
                                                        const PadStrideInfo &pad_stride_info,
                                                        const Size2D        &dilation = Size2D(1U, 1U));

/** Returns calculated width and height of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width           Width of input tensor (Number of columns)
 * @param[in] height          Height of input tensor (Number of rows)
 * @param[in] kernel_width    Kernel width.
 * @param[in] kernel_height   Kernel height.
 * @param[in] pad_stride_info Pad and stride information.
 *
 * @return A pair with the new width in the first position and the new height in the second, returned values can be < 1
 */
std::pair<int, int> scaled_dimensions_signed(
    int width, int height, int kernel_width, int kernel_height, const PadStrideInfo &pad_stride_info);

/** Returns calculated width, height and depth of output scaled tensor depending on dimensions rounding mode.
 *
 * @param[in] width         Width of input tensor
 * @param[in] height        Height of input tensor
 * @param[in] depth         Depth of input tensor
 * @param[in] kernel_width  Kernel width.
 * @param[in] kernel_height Kernel height.
 * @param[in] kernel_depth  Kernel depth.
 * @param[in] pool3d_info   Pad and stride and round information for 3d pooling
 *
 * @return A tuple with the new width in the first position, the new height in the second, and the new depth in the third.
 *         Returned values can be < 1
 */
std::tuple<int, int, int> scaled_3d_dimensions_signed(int                       width,
                                                      int                       height,
                                                      int                       depth,
                                                      int                       kernel_width,
                                                      int                       kernel_height,
                                                      int                       kernel_depth,
                                                      const Pooling3dLayerInfo &pool3d_info);

/** Check if the given reduction operation should be handled in a serial way.
 *
 * @param[in] op   Reduction operation to perform
 * @param[in] dt   Data type
 * @param[in] axis Axis along which to reduce
 *
 * @return True if the given reduction operation should be handled in a serial way.
 */
bool needs_serialized_reduction(ReductionOperation op, DataType dt, unsigned int axis);

/** Returns output quantization information for softmax layer
 *
 * @param[in] input_type The data type of the input tensor
 * @param[in] is_log     True for log softmax
 *
 * @return Quantization information for the output tensor
 */
QuantizationInfo get_softmax_output_quantization_info(DataType input_type, bool is_log);

/** Returns a pair of minimum and maximum values for a quantized activation
 *
 * @param[in] act_info  The information for activation
 * @param[in] data_type The used data type
 * @param[in] oq_info   The output quantization information
 *
 * @return The pair with minimum and maximum values
 */
std::pair<int32_t, int32_t> get_quantized_activation_min_max(const ActivationLayerInfo &act_info,
                                                             DataType                   data_type,
                                                             UniformQuantizationInfo    oq_info);

/** Convert a channel identity into a string.
 *
 * @param[in] channel @ref Channel to be translated to string.
 *
 * @return The string describing the channel.
 */
const std::string &string_from_channel(Channel channel);

/** Translates a given border mode policy to a string.
 *
 * @param[in] border_mode @ref BorderMode to be translated to string.
 *
 * @return The string describing the border mode.
 */
const std::string &string_from_border_mode(BorderMode border_mode);
/** Translates a given normalization type to a string.
 *
 * @param[in] type @ref NormType to be translated to string.
 *
 * @return The string describing the normalization type.
 */
const std::string &string_from_norm_type(NormType type);
/** Translates a given pooling type to a string.
 *
 * @param[in] type @ref PoolingType to be translated to string.
 *
 * @return The string describing the pooling type.
 */
const std::string &string_from_pooling_type(PoolingType type);
/** Check if the pool region is entirely outside the input tensor
 *
 * @param[in] info @ref PoolingLayerInfo to be checked.
 *
 * @return True if the pool region is entirely outside the input tensor, False otherwise.
 */
bool is_pool_region_entirely_outside_input(const PoolingLayerInfo &info);
/** Check if the 3d pool region is entirely outside the input tensor
 *
 * @param[in] info @ref Pooling3dLayerInfo to be checked.
 *
 * @return True if the pool region is entirely outside the input tensor, False otherwise.
 */
bool is_pool_3d_region_entirely_outside_input(const Pooling3dLayerInfo &info);
/** Check if the 3D padding is symmetric i.e. padding in each opposite sides are euqal (left=right, top=bottom and front=back)
 *
 * @param[in] info @ref Padding3D input 3D padding object to check if it is symmetric
 *
 * @return True if padding is symmetric
 */
inline bool is_symmetric(const Padding3D &info)
{
    return ((info.left == info.right) && (info.top == info.bottom) && (info.front == info.back));
}
/** Translates a given GEMMLowp output stage to a string.
 *
 * @param[in] output_stage @ref GEMMLowpOutputStageInfo to be translated to string.
 *
 * @return The string describing the GEMMLowp output stage
 */
const std::string &string_from_gemmlowp_output_stage(GEMMLowpOutputStageType output_stage);
/** Convert a PixelValue to a string, represented through the specific data type
 *
 * @param[in] value     The PixelValue to convert
 * @param[in] data_type The type to be used to convert the @p value
 *
 * @return String representation of the PixelValue through the given data type.
 */
std::string string_from_pixel_value(const PixelValue &value, const DataType data_type);

/** Stores padding information before configuring a kernel
 *
 * @param[in] infos list of tensor infos to store the padding info for
 *
 * @return An unordered map where each tensor info pointer is paired with its original padding info
 */
std::unordered_map<const ITensorInfo *, PaddingSize> get_padding_info(std::initializer_list<const ITensorInfo *> infos);
/** Stores padding information before configuring a kernel
 *
 * @param[in] tensors list of tensors to store the padding info for
 *
 * @return An unordered map where each tensor info pointer is paired with its original padding info
 */
std::unordered_map<const ITensorInfo *, PaddingSize> get_padding_info(std::initializer_list<const ITensor *> tensors);
/** Check if the previously stored padding info has changed after configuring a kernel
 *
 * @param[in] padding_map an unordered map where each tensor info pointer is paired with its original padding info
 *
 * @return true if any of the tensor infos has changed its paddings
 */
bool has_padding_changed(const std::unordered_map<const ITensorInfo *, PaddingSize> &padding_map);

/** Returns the number of elements required to go from start to end with the wanted step
 *
 * @param[in] start start value
 * @param[in] end   end value
 * @param[in] step  step value between each number in the wanted sequence
 *
 * @return number of elements to go from start value to end value using the wanted step
 */
inline size_t num_of_elements_in_range(const float start, const float end, const float step)
{
    ARM_COMPUTE_ERROR_ON_MSG(step == 0, "Range Step cannot be 0");
    return size_t(std::ceil((end - start) / step));
}

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
/** Print consecutive elements to an output stream.
 *
 * @param[out] s             Output stream to print the elements to.
 * @param[in]  ptr           Pointer to print the elements from.
 * @param[in]  n             Number of elements to print.
 * @param[in]  stream_width  (Optional) Width of the stream. If set to 0 the element's width is used. Defaults to 0.
 * @param[in]  element_delim (Optional) Delimeter among the consecutive elements. Defaults to space delimeter
 */
template <typename T>
void print_consecutive_elements_impl(
    std::ostream &s, const T *ptr, unsigned int n, int stream_width = 0, const std::string &element_delim = " ")
{
    using print_type = typename std::conditional<std::is_floating_point<T>::value, T, int>::type;
    std::ios stream_status(nullptr);
    stream_status.copyfmt(s);

    for (unsigned int i = 0; i < n; ++i)
    {
        // Set stream width as it is not a "sticky" stream manipulator
        if (stream_width != 0)
        {
            s.width(stream_width);
        }

        if (std::is_same<typename std::decay<T>::type, half>::value)
        {
            // We use T instead of print_type here is because the std::is_floating_point<half> returns false and then the print_type becomes int.
            s << std::right << static_cast<T>(ptr[i]) << element_delim;
        }
        else if (std::is_same<typename std::decay<T>::type, bfloat16>::value)
        {
            // We use T instead of print_type here is because the std::is_floating_point<bfloat16> returns false and then the print_type becomes int.
            s << std::right << float(ptr[i]) << element_delim;
        }
        else
        {
            s << std::right << static_cast<print_type>(ptr[i]) << element_delim;
        }
    }

    // Restore output stream flags
    s.copyfmt(stream_status);
}

/** Identify the maximum width of n consecutive elements.
 *
 * @param[in] s   The output stream which will be used to print the elements. Used to extract the stream format.
 * @param[in] ptr Pointer to the elements.
 * @param[in] n   Number of elements.
 *
 * @return The maximum width of the elements.
 */
template <typename T>
int max_consecutive_elements_display_width_impl(std::ostream &s, const T *ptr, unsigned int n)
{
    using print_type = typename std::conditional<std::is_floating_point<T>::value, T, int>::type;

    int max_width = -1;
    for (unsigned int i = 0; i < n; ++i)
    {
        std::stringstream ss;
        ss.copyfmt(s);

        if (std::is_same<typename std::decay<T>::type, half>::value)
        {
            // We use T instead of print_type here is because the std::is_floating_point<half> returns false and then the print_type becomes int.
            ss << static_cast<T>(ptr[i]);
        }
        else if (std::is_same<typename std::decay<T>::type, bfloat16>::value)
        {
            // We use T instead of print_type here is because the std::is_floating_point<bfloat> returns false and then the print_type becomes int.
            ss << float(ptr[i]);
        }
        else
        {
            ss << static_cast<print_type>(ptr[i]);
        }

        max_width = std::max<int>(max_width, ss.str().size());
    }
    return max_width;
}

/** Print consecutive elements to an output stream.
 *
 * @param[out] s             Output stream to print the elements to.
 * @param[in]  dt            Data type of the elements
 * @param[in]  ptr           Pointer to print the elements from.
 * @param[in]  n             Number of elements to print.
 * @param[in]  stream_width  (Optional) Width of the stream. If set to 0 the element's width is used. Defaults to 0.
 * @param[in]  element_delim (Optional) Delimeter among the consecutive elements. Defaults to space delimeter
 */
void print_consecutive_elements(std::ostream      &s,
                                DataType           dt,
                                const uint8_t     *ptr,
                                unsigned int       n,
                                int                stream_width,
                                const std::string &element_delim = " ");

/** Identify the maximum width of n consecutive elements.
 *
 * @param[in] s   Output stream to print the elements to.
 * @param[in] dt  Data type of the elements
 * @param[in] ptr Pointer to print the elements from.
 * @param[in] n   Number of elements to print.
 *
 * @return The maximum width of the elements.
 */
int max_consecutive_elements_display_width(std::ostream &s, DataType dt, const uint8_t *ptr, unsigned int n);
#endif /* ARM_COMPUTE_ASSERTS_ENABLED */
} // namespace arm_compute
#endif /*ARM_COMPUTE_UTILS_H */
