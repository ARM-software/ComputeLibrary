/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_HELPERS_H
#define ACL_TESTS_VALIDATION_HELPERS_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "support/Half.h"
#include "tests/Globals.h"
#include "tests/SimpleTensor.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename T>
struct is_floating_point : public std::is_floating_point<T>
{
};

template <>
struct is_floating_point<half> : public std::true_type
{
};

/** Helper struct to store the hints for
 *  - destination quantization info
 *  - minimum bias value
 *  - maximum bias value
 * in quantized test construction.
 */
struct QuantizationHint
{
    QuantizationInfo q_info;
    int32_t          bias_min;
    int32_t          bias_max;
};

/** Helper function to get the testing range for each activation layer.
 *
 * @param[in] activation Activation function to test.
 * @param[in] data_type  Data type.
 *
 * @return A pair containing the lower upper testing bounds for a given function.
 */
template <typename T>
std::pair<T, T> get_activation_layer_test_bounds(ActivationLayerInfo::ActivationFunction activation, DataType data_type)
{
    std::pair<T, T> bounds;

    switch(data_type)
    {
        case DataType::F16:
        {
            using namespace half_float::literal;

            switch(activation)
            {
                case ActivationLayerInfo::ActivationFunction::TANH:
                case ActivationLayerInfo::ActivationFunction::SQUARE:
                case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                    // Reduce range as exponent overflows
                    bounds = std::make_pair(-2._h, 2._h);
                    break;
                case ActivationLayerInfo::ActivationFunction::SQRT:
                    // Reduce range as sqrt should take a non-negative number
                    bounds = std::make_pair(0._h, 128._h);
                    break;
                default:
                    bounds = std::make_pair(-255._h, 255._h);
                    break;
            }
            break;
        }
        case DataType::F32:
            switch(activation)
            {
                case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                    // Reduce range as exponent overflows
                    bounds = std::make_pair(-40.f, 40.f);
                    break;
                case ActivationLayerInfo::ActivationFunction::SQRT:
                    // Reduce range as sqrt should take a non-negative number
                    bounds = std::make_pair(0.f, 255.f);
                    break;
                default:
                    bounds = std::make_pair(-255.f, 255.f);
                    break;
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }

    return bounds;
}

/** Convert an asymmetric quantized simple tensor into float using tensor quantization information.
 *
 * @param[in] src Quantized tensor.
 *
 * @return Float tensor.
 */
template <typename T>
SimpleTensor<float> convert_from_asymmetric(const SimpleTensor<T> &src);

/** Convert float simple tensor into quantized using specified quantization information.
 *
 * @param[in] src               Float tensor.
 * @param[in] quantization_info Quantification information.
 *
 * \relates  arm_compute::test::SimpleTensor
 * @return Quantized tensor.
 */
template <typename T>
SimpleTensor<T> convert_to_asymmetric(const SimpleTensor<float> &src, const QuantizationInfo &quantization_info);

/** Convert quantized simple tensor into float using tensor quantization information.
 *
 * @param[in] src Quantized tensor.
 *
 * @return Float tensor.
 */
template <typename T>
SimpleTensor<float> convert_from_symmetric(const SimpleTensor<T> &src);

/** Convert float simple tensor into quantized using specified quantization information.
 *
 * @param[in] src               Float tensor.
 * @param[in] quantization_info Quantification information.
 * \relates  arm_compute::test::SimpleTensor
 * @return Quantized tensor.
 */
template <typename T>
SimpleTensor<T> convert_to_symmetric(const SimpleTensor<float> &src, const QuantizationInfo &quantization_info);

/** Matrix multiply between 2 float simple tensors
 *
 * @param[in]  a   Input tensor A
 * @param[in]  b   Input tensor B
 * @param[out] out Output tensor
 *
 */
template <typename T>
void matrix_multiply(const SimpleTensor<T> &a, const SimpleTensor<T> &b, SimpleTensor<T> &out);

/** Transpose matrix
 *
 * @param[in]  in  Input tensor
 * @param[out] out Output tensor
 *
 */
template <typename T>
void transpose_matrix(const SimpleTensor<T> &in, SimpleTensor<T> &out);

/** Get a 2D tile from a tensor
 *
 * @note In case of out-of-bound reads, the tile will be filled with zeros
 *
 * @param[in]  in    Input tensor
 * @param[out] tile  Tile
 * @param[in]  coord Coordinates
 */
template <typename T>
void get_tile(const SimpleTensor<T> &in, SimpleTensor<T> &tile, const Coordinates &coord);

/** Fill with zeros the input tensor in the area defined by anchor and shape
 *
 * @param[in]  in     Input tensor to fill with zeros
 * @param[out] anchor Starting point of the zeros area
 * @param[in]  shape  Ending point of the zeros area
 */
template <typename T>
void zeros(SimpleTensor<T> &in, const Coordinates &anchor, const TensorShape &shape);

/** Helper function to compute quantized min and max bounds
 *
 * @param[in] quant_info Quantization info to be used for conversion
 * @param[in] min        Floating point minimum value to be quantized
 * @param[in] max        Floating point maximum value to be quantized
 */
std::pair<int, int> get_quantized_bounds(const QuantizationInfo &quant_info, float min, float max);

/** Helper function to compute asymmetric quantized signed min and max bounds
 *
 * @param[in] quant_info Quantization info to be used for conversion
 * @param[in] min        Floating point minimum value to be quantized
 * @param[in] max        Floating point maximum value to be quantized
 */
std::pair<int, int> get_quantized_qasymm8_signed_bounds(const QuantizationInfo &quant_info, float min, float max);

/** Helper function to compute symmetric quantized min and max bounds
 *
 * @param[in] quant_info Quantization info to be used for conversion
 * @param[in] min        Floating point minimum value to be quantized
 * @param[in] max        Floating point maximum value to be quantized
 * @param[in] channel_id Channel id for per channel quantization info.
 */
std::pair<int, int> get_symm_quantized_per_channel_bounds(const QuantizationInfo &quant_info, float min, float max, size_t channel_id = 0);

/** Add random padding along the X axis (between 1 and 16 columns per side) to all the input tensors.
 *  This is used in our validation suite in order to simulate implicit padding addition after configuring, but before allocating.
 *
 * @param[in] tensors        List of tensors to add padding to
 * @param[in] data_layout    (Optional) Data layout of the operator
 * @param[in] only_right_pad (Optional) Only right padding testing, in case of cl image padding
 *
 * @note This function adds padding to the input tensors only if data_layout == DataLayout::NHWC
 */
void add_padding_x(std::initializer_list<ITensor *> tensors, const DataLayout &data_layout = DataLayout::NHWC, bool only_right_pad = false);

/** For 2d convolution, given the Lhs/Rhs matrix quantization informations and the convolution dimension,
 *  calculate a suitable output quantization and suggested bias range for obtaining non-saturated outputs with high probability.
 *
 * @param[in] in_q_info     Input matrix quantization info
 * @param[in] weight_q_info Weights matrix quantization info
 * @param[in] height        Height of the weights tensor
 * @param[in] width         Width of the weights tensors
 * @param[in] channels      Number of input channels
 * @param[in] data_type     data type, only QASYMM8, QASYMM8_SIGNED are supported
 * @param[in] bias_fraction see @ref suggest_mac_dst_q_info_and_bias() for explanation
 *
 * @return QuantizationHint object containing the suggested output quantization info and min/max bias range
 */
QuantizationHint suggest_conv_dst_q_info_and_bias(const QuantizationInfo &in_q_info,
                                                  const QuantizationInfo &weight_q_info,
                                                  int32_t height,
                                                  int32_t width,
                                                  int32_t channels,
                                                  DataType data_type,
                                                  float bias_fraction);

/** For a matrix multiplication, given the Lhs/Rhs matrix quantization informations and the matrix multiplication dimensions,
 *  calculate a suitable output quantization and suggested bias range for obtaining non-saturated outputs with high probability.
 *
 * @param[in] lhs_q_info    Lhs matrix quantization info
 * @param[in] rhs_q_info    Rhs matrix quantization info
 * @param[in] m             Number of rows of Lhs matrix
 * @param[in] n             Number of columns of Rhs Matrix
 * @param[in] k             Number of rows/columns of Rhs/Lhs Matrix
 * @param[in] data_type     data type, only QASYMM8, QASYMM8_SIGNED are supported
 * @param[in] bias_fraction see @ref suggest_mac_dst_q_info_and_bias() for explanation
 *
 * @return QuantizationHint object containing the suggested output quantization info and min/max bias range
 */
QuantizationHint suggest_matmul_dst_q_info_and_bias(const QuantizationInfo &lhs_q_info,
                                                    const QuantizationInfo &rhs_q_info, int32_t m, int32_t n, int32_t k, DataType data_type,
                                                    float bias_fraction);

/** For a multiply-accumulate (mac), given the Lhs/Rhs vector quantization informations and the dot product dimensions,
 *  calculate a suitable output quantization and suggested bias range for obtaining non-saturated outputs with high probability.
 *
 * @param[in] lhs_q_info    Lhs matrix quantization info
 * @param[in] rhs_q_info    Rhs matrix quantization info
 * @param[in] k             number of accumulations taking place in the sum, i.e. c_k = sum_k(a_k * b_k)
 * @param[in] data_type     data type, only QASYMM8, QASYMM8_SIGNED are supported
 * @param[in] bias_fraction the fraction of bias amplitude compared to integer accummulation.
 * @param[in] num_sd        (Optional) number of standard deviations we allow from the mean. Default value is 2.
 *
 * @return QuantizationHint object containing the suggested output quantization info and min/max bias range
 */
QuantizationHint suggest_mac_dst_q_info_and_bias(const QuantizationInfo &lhs_q_info,
                                                 const QuantizationInfo &rhs_q_info, int32_t k, DataType data_type, float bias_fraction,
                                                 int num_sd = 2);
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_HELPERS_H
