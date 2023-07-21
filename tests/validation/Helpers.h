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
#ifndef ACL_TESTS_VALIDATION_HELPERS
#define ACL_TESTS_VALIDATION_HELPERS

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "support/Half.h"
#include "tests/Globals.h"
#include "tests/SimpleTensor.h"

#include <math.h>
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

/** Calculate output tensor shape give a vector of input tensor to concatenate
 *
 * @param[in] input_shapes Shapes of the tensors to concatenate across depth.
 *
 * @return The shape of output concatenated tensor.
 */
TensorShape calculate_depth_concatenate_shape(const std::vector<TensorShape> &input_shapes);

/** Calculate output tensor shape for the concatenate operation along a given axis
 *
 * @param[in] input_shapes Shapes of the tensors to concatenate across width.
 * @param[in] axis         Axis to use for the concatenate operation
 *
 * @return The shape of output concatenated tensor.
 */
TensorShape calculate_concatenate_shape(const std::vector<TensorShape> &input_shapes, size_t axis);

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

/** Add random padding along the Y axis (between 1 and 4 rows per side) to all the input tensors.
 *  This is used in our validation suite in order to simulate implicit padding addition after configuring, but before allocating.
 *
 * @param[in] tensors     List of tensors to add padding to
 * @param[in] data_layout (Optional) Data layout of the operator
 *
 * @note This function adds padding to the input tensors only if data_layout == DataLayout::NHWC
 */
void add_padding_y(std::initializer_list<ITensor *> tensors, const DataLayout &data_layout = DataLayout::NHWC);

/** For MatMulLowp, given the Lhs/Rhs matrix quantization informations and the matrix multiplication dimensions,
 *  calculate a suitable output quantization for obtaining non-saturated outputs with high probability.
 */
QuantizationInfo calculate_mat_mul_dst_q_info(const QuantizationInfo &lhs_q_info, const QuantizationInfo &rhs_q_info, int m, int n, int k, DataType data_type);
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ACL_TESTS_VALIDATION_HELPERS */
