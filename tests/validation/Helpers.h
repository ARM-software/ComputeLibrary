/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_VALIDATION_HELPERS_H__
#define __ARM_COMPUTE_TEST_VALIDATION_HELPERS_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "support/Half.h"
#include "tests/Globals.h"
#include "tests/SimpleTensor.h"

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
                case ActivationLayerInfo::ActivationFunction::LOGISTIC:
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

/** Fill mask with the corresponding given pattern.
 *
 * @param[in,out] mask    Mask to be filled according to pattern
 * @param[in]     cols    Columns (width) of mask
 * @param[in]     rows    Rows (height) of mask
 * @param[in]     pattern Pattern to fill the mask according to
 */
void fill_mask_from_pattern(uint8_t *mask, int cols, int rows, MatrixPattern pattern);

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

/** Parameters of Harris Corners algorithm. */
struct HarrisCornersParameters
{
    float   threshold{ 0.f };           /**< Threshold */
    float   sensitivity{ 0.f };         /**< Sensitivity */
    float   min_dist{ 0.f };            /**< Minimum distance */
    uint8_t constant_border_value{ 0 }; /**< Border value */
};

/** Generate parameters for Harris Corners algorithm. */
HarrisCornersParameters harris_corners_parameters();

/** Parameters of Canny edge algorithm. */
struct CannyEdgeParameters
{
    int32_t upper_thresh{ 255 };
    int32_t lower_thresh{ 0 };
    uint8_t constant_border_value{ 0 };
};

/** Generate parameters for Canny edge algorithm. */
CannyEdgeParameters canny_edge_parameters();

/** Helper function to fill the Lut random by a ILutAccessor.
 *
 * @param[in,out] table Accessor at the Lut.
 *
 */
template <typename T>
void fill_lookuptable(T &&table)
{
    std::mt19937                                          generator(library->seed());
    std::uniform_int_distribution<typename T::value_type> distribution(std::numeric_limits<typename T::value_type>::min(), std::numeric_limits<typename T::value_type>::max());

    for(int i = std::numeric_limits<typename T::value_type>::min(); i <= std::numeric_limits<typename T::value_type>::max(); i++)
    {
        table[i] = distribution(generator);
    }
}

/** Convert quantized simple tensor into float using tensor quantization information.
 *
 * @param[in] src Quantized tensor.
 *
 * @return Float tensor.
 */
SimpleTensor<float> convert_from_asymmetric(const SimpleTensor<uint8_t> &src);

/** Convert float simple tensor into quantized using specified quantization information.
 *
 * @param[in] src               Float tensor.
 * @param[in] quantization_info Quantification information.
 *
 * @return Quantized tensor.
 */
SimpleTensor<uint8_t> convert_to_asymmetric(const SimpleTensor<float> &src, const QuantizationInfo &quantization_info);

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
 *
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
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_VALIDATION_HELPERS_H__ */
