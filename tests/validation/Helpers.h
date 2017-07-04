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
#include "tests/Globals.h"
#include "tests/ILutAccessor.h"
#include "tests/Types.h"
#include "tests/validation/ValidationUserConfiguration.h"
#include "tests/validation/half.h"

#include <array>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Helper function to fill one or more tensors with the uniform distribution with int values.
 *
 * @param[in]     dist          Distribution to be used to get the values for the tensor.
 * @param[in]     seeds         List of seeds to be used to fill each tensor.
 * @param[in,out] tensor        Tensor to be initialized with the values of the distribution.
 * @param[in,out] other_tensors (Optional) One or more tensors to be filled.
 *
 */
template <typename D, typename T, typename... Ts>
void fill_tensors(D &&dist, std::initializer_list<int> seeds, T &&tensor, Ts &&... other_tensors)
{
    const std::array < T, 1 + sizeof...(Ts) > tensors{ { std::forward<T>(tensor), std::forward<Ts>(other_tensors)... } };
    std::vector<int> vs(seeds);
    ARM_COMPUTE_ERROR_ON(vs.size() != tensors.size());
    int k = 0;
    for(auto tp : tensors)
    {
        library->fill(*tp, std::forward<D>(dist), vs[k++]);
    }
}

/** Helper function to get the testing range for each activation layer.
 *
 * @param[in] activation           Activation function to test.
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part. Defaults to 1.
 *
 * @return A pair containing the lower upper testing bounds for a given function.
 */
template <typename T>
inline std::pair<T, T> get_activation_layer_test_bounds(ActivationLayerInfo::ActivationFunction activation, int fixed_point_position = 1)
{
    bool is_float = std::is_same<T, float>::value;
    is_float      = is_float || std::is_same<T, half_float::half>::value;

    std::pair<T, T> bounds;

    // Set initial values
    if(is_float)
    {
        bounds = std::make_pair(-255.f, 255.f);
    }
    else
    {
        bounds = std::make_pair(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
    }

    // Reduce testing ranges
    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            // Reduce range as exponent overflows
            if(is_float)
            {
                bounds.first  = -40.f;
                bounds.second = 40.f;
            }
            else
            {
                bounds.first  = -(1 << (fixed_point_position));
                bounds.second = 1 << (fixed_point_position);
            }
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
            // Reduce range as exponent overflows
            if(!is_float)
            {
                bounds.first  = -(1 << (fixed_point_position));
                bounds.second = 1 << (fixed_point_position);
            }
            break;
        case ActivationLayerInfo::ActivationFunction::SQRT:
            // Reduce range as sqrt should take a non-negative number
            bounds.first = (is_float) ? 0 : 1;
            break;
        default:
            break;
    }
    return bounds;
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
    bool is_float = std::is_floating_point<T>::value;
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

/** Fill mask with the corresponding given pattern.
 *
 * @param[in,out] mask    Mask to be filled according to pattern
 * @param[in]     cols    Columns (width) of mask
 * @param[in]     rows    Rows (height) of mask
 * @param[in]     pattern Pattern to fill the mask according to
 */
inline void fill_mask_from_pattern(uint8_t *mask, int cols, int rows, MatrixPattern pattern)
{
    unsigned int                v = 0;
    std::mt19937                gen(user_config.seed.get());
    std::bernoulli_distribution dist(0.5);

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < cols; ++c, ++v)
        {
            uint8_t val = 0;

            switch(pattern)
            {
                case MatrixPattern::BOX:
                    val = 255;
                    break;
                case MatrixPattern::CROSS:
                    val = ((r == (rows / 2)) || (c == (cols / 2))) ? 255 : 0;
                    break;
                case MatrixPattern::DISK:
                    val = (((r - rows / 2.0f + 0.5f) * (r - rows / 2.0f + 0.5f)) / ((rows / 2.0f) * (rows / 2.0f)) + ((c - cols / 2.0f + 0.5f) * (c - cols / 2.0f + 0.5f)) / ((cols / 2.0f) *
                            (cols / 2.0f))) <= 1.0f ? 255 : 0;
                    break;
                case MatrixPattern::OTHER:
                    val = (dist(gen) ? 0 : 255);
                    break;
                default:
                    return;
            }

            mask[v] = val;
        }
    }

    if(pattern == MatrixPattern::OTHER)
    {
        std::uniform_int_distribution<uint8_t> distribution_u8(0, ((cols * rows) - 1));
        mask[distribution_u8(gen)] = 255;
    }
}

/** Calculate output tensor shape give a vector of input tensor to concatenate
 *
 * @param[in] input_shapes Shapes of the tensors to concatenate across depth.
 *
 * @return The shape of output concatenated tensor.
 */
inline TensorShape calculate_depth_concatenate_shape(std::vector<TensorShape> input_shapes)
{
    TensorShape out_shape = input_shapes.at(0);

    unsigned int max_x = 0;
    unsigned int max_y = 0;
    unsigned int depth = 0;

    for(auto const &shape : input_shapes)
    {
        max_x = std::max<unsigned int>(shape.x(), max_x);
        max_y = std::max<unsigned int>(shape.y(), max_y);
        depth += shape.z();
    }

    out_shape.set(0, max_x);
    out_shape.set(1, max_y);
    out_shape.set(2, depth);

    return out_shape;
}

/** Fill matrix random.
 *
 * @param[in,out] matrix Matrix
 * @param[in]     cols   Columns (width) of matrix
 * @param[in]     rows   Rows (height) of matrix
 */
template <std::size_t SIZE>
inline void fill_warp_matrix(std::array<float, SIZE> &matrix, int cols, int rows)
{
    std::mt19937                          gen(user_config.seed.get());
    std::uniform_real_distribution<float> dist(-1, 1);

    for(int v = 0, r = 0; r < rows; ++r)
    {
        for(int c = 0; c < cols; ++c, ++v)
        {
            matrix[v] = dist(gen);
        }
    }
    if(SIZE == 9)
    {
        matrix[(cols * rows) - 1] = 1;
    }
}

/** Create a vector of random ROIs.
 *
 * @param[in] shape     The shape of the input tensor.
 * @param[in] pool_info The ROI pooling information.
 * @param[in] num_rois  The number of ROIs to be created.
 * @param[in] seed      The random seed to be used.
 *
 * @return A vector that contains the requested number of random ROIs
 */
std::vector<ROI> generate_random_rois(const TensorShape &shape, const ROIPoolingLayerInfo &pool_info, unsigned int num_rois, std::random_device::result_type seed);

/** Helper function to fill the Lut random by a ILutAccessor.
 *
 * @param[in,out] table Accessor at the Lut.
 *
 */
template <typename T>
void fill_lookuptable(T &&table)
{
    std::mt19937                                          generator(user_config.seed.get());
    std::uniform_int_distribution<typename T::value_type> distribution(std::numeric_limits<typename T::value_type>::min(), std::numeric_limits<typename T::value_type>::max());

    for(int i = std::numeric_limits<typename T::value_type>::min(); i <= std::numeric_limits<typename T::value_type>::max(); i++)
    {
        table[i] = distribution(generator);
    }
}

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_VALIDATION_HELPERS_H__ */
