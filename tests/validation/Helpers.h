/*
 * Copyright (c) 2017 ARM Limited.
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
 * @param[in] activation           Activation function to test.
 * @param[in] data_type            Data type.
 * @param[in] fixed_point_position Number of bits for the fractional part. Defaults to 1.
 *
 * @return A pair containing the lower upper testing bounds for a given function.
 */
template <typename T>
std::pair<T, T> get_activation_layer_test_bounds(ActivationLayerInfo::ActivationFunction activation, DataType data_type, int fixed_point_position = 0)
{
    std::pair<T, T> bounds;

    switch(data_type)
    {
        case DataType::F16:
        {
            using namespace half_float::literal;

            switch(activation)
            {
                case ActivationLayerInfo::ActivationFunction::SQUARE:
                case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                    // Reduce range as exponent overflows
                    bounds = std::make_pair(-10._h, 10._h);
                    break;
                case ActivationLayerInfo::ActivationFunction::SQRT:
                    // Reduce range as sqrt should take a non-negative number
                    bounds = std::make_pair(0._h, 255._h);
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
        case DataType::QS8:
        case DataType::QS16:
            switch(activation)
            {
                case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                case ActivationLayerInfo::ActivationFunction::TANH:
                    // Reduce range as exponent overflows
                    bounds = std::make_pair(-(1 << fixed_point_position), 1 << fixed_point_position);
                    break;
                case ActivationLayerInfo::ActivationFunction::SQRT:
                    // Reduce range as sqrt should take a non-negative number
                    // Can't be zero either as inv_sqrt is used in NEON.
                    bounds = std::make_pair(1, std::numeric_limits<T>::max());
                    break;
                default:
                    bounds = std::make_pair(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
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

/** Parameters of Harris Corners algorithm. */
struct HarrisCornersParameters
{
    float   threshold{ 0.f };
    float   sensitivity{ 0.f };
    float   min_dist{ 0.f };
    uint8_t constant_border_value{ 0 };
};

/** Generate parameters for Harris Corners algorithm. */
HarrisCornersParameters harris_corners_parameters();

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

/** Helper function to get the testing range for batch normalization layer.
 *
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part. Defaults to 1.
 *
 * @return A pair containing the lower upper testing bounds.
 */
template <typename T>
std::pair<T, T> get_batchnormalization_layer_test_bounds(int fixed_point_position = 1)
{
    const bool is_float = std::is_floating_point<T>::value;
    std::pair<T, T> bounds;

    // Set initial values
    if(is_float)
    {
        bounds = std::make_pair(-1.f, 1.f);
    }
    else
    {
        bounds = std::make_pair(1, 1 << (fixed_point_position));
    }

    return bounds;
}

/** Helper function to get the testing range for NormalizePlanarYUV layer.
 *
 * @return A pair containing the lower upper testing bounds.
 */
template <typename T>
std::pair<T, T> get_normalize_planar_yuv_layer_test_bounds()
{
    std::pair<T, T> bounds;

    bounds = std::make_pair(-1.f, 1.f);

    return bounds;
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
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_VALIDATION_HELPERS_H__ */
