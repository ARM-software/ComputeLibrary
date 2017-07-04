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
#include "Reference.h"

#include "AssetsLibrary.h"
#include "Globals.h"
#include "Helpers.h"
#include "ReferenceCPP.h"
#include "validation/Helpers.h"

#include <random>
#include <vector>

using namespace arm_compute::test;

#ifndef DOXYGEN_SKIP_THIS
namespace arm_compute
{
namespace test
{
namespace validation
{
std::pair<RawTensor, RawTensor> Reference::compute_reference_sobel_3x3(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create reference
    RawTensor ref_src(shape, Format::U8);
    RawTensor ref_dst_x(shape, Format::S16);
    RawTensor ref_dst_y(shape, Format::S16);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::sobel_3x3(ref_src, ref_dst_x, ref_dst_y, border_mode, constant_border_value);

    return std::make_pair(ref_dst_x, ref_dst_y);
}
std::pair<RawTensor, RawTensor> Reference::compute_reference_sobel_5x5(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create reference
    RawTensor ref_src(shape, Format::U8);
    RawTensor ref_dst_x(shape, Format::S16);
    RawTensor ref_dst_y(shape, Format::S16);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::sobel_5x5(ref_src, ref_dst_x, ref_dst_y, border_mode, constant_border_value);

    return std::make_pair(ref_dst_x, ref_dst_y);
}
void Reference::compute_reference_min_max_location(const TensorShape &shape, DataType dt_in, void *min, void *max, IArray<Coordinates2D> &min_loc, IArray<Coordinates2D> &max_loc,
                                                   uint32_t &min_count, uint32_t &max_count)
{
    // Create reference
    RawTensor ref_src(shape, dt_in);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::min_max_location(ref_src, min, max, min_loc, max_loc, min_count, max_count);
}
std::pair<float, float> Reference::compute_reference_mean_and_standard_deviation(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);

    // Create output variables
    float mean;
    float std_dev;

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::mean_and_standard_deviation(ref_src, mean, std_dev);

    return std::make_pair(mean, std_dev);
}
RawTensor Reference::compute_reference_integral_image(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U32);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::integral_image(ref_src, ref_dst);

    return ref_dst;
}
RawTensor Reference::compute_reference_absolute_difference(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out)
{
    // Create reference
    RawTensor ref_src1(shape, dt_in0);
    RawTensor ref_src2(shape, dt_in1);
    RawTensor ref_dst(shape, dt_out);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::absolute_difference(ref_src1, ref_src2, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_accumulate(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::S16);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);
    library->fill_tensor_uniform(ref_dst, 1);

    // Compute reference
    ReferenceCPP::accumulate(ref_src, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_accumulate_squared(const TensorShape &shape, uint32_t shift)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::S16);

    // Fill reference
    // ref_dst tensor filled with non-negative values
    library->fill_tensor_uniform(ref_src, 0);
    library->fill_tensor_uniform(ref_dst, 1, static_cast<int16_t>(0), std::numeric_limits<int16_t>::max());

    // Compute reference
    ReferenceCPP::accumulate_squared(ref_src, ref_dst, shift);

    return ref_dst;
}

RawTensor Reference::compute_reference_accumulate_weighted(const TensorShape &shape, float alpha)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);
    library->fill_tensor_uniform(ref_dst, 1);

    // Compute reference
    ReferenceCPP::accumulate_weighted(ref_src, ref_dst, alpha);

    return ref_dst;
}

RawTensor Reference::compute_reference_arithmetic_addition(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, ConvertPolicy convert_policy, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src1(shape, dt_in0, 1, fixed_point_position);
    RawTensor ref_src2(shape, dt_in1, 1, fixed_point_position);
    RawTensor ref_dst(shape, dt_out, 1, fixed_point_position);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::arithmetic_addition(ref_src1, ref_src2, ref_dst, convert_policy);

    return ref_dst;
}

RawTensor Reference::compute_reference_arithmetic_subtraction(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, ConvertPolicy convert_policy, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src1(shape, dt_in0, 1, fixed_point_position);
    RawTensor ref_src2(shape, dt_in1, 1, fixed_point_position);
    RawTensor ref_dst(shape, dt_out, 1, fixed_point_position);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::arithmetic_subtraction(ref_src1, ref_src2, ref_dst, convert_policy);

    return ref_dst;
}

RawTensor Reference::compute_reference_bitwise_and(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src1(shape, DataType::U8);
    RawTensor ref_src2(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::bitwise_and(ref_src1, ref_src2, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_bitwise_or(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src1(shape, DataType::U8);
    RawTensor ref_src2(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::bitwise_or(ref_src1, ref_src2, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_bitwise_xor(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src1(shape, DataType::U8);
    RawTensor ref_src2(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::bitwise_xor(ref_src1, ref_src2, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_bitwise_not(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::bitwise_not(ref_src, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_box3x3(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::box3x3(ref_src, ref_dst, border_mode, constant_border_value);

    return ref_dst;
}

RawTensor Reference::compute_reference_depth_convert(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy,
                                                     uint32_t shift, uint32_t fixed_point_position_in, uint32_t fixed_point_position_out)
{
    RawTensor ref_src(shape, dt_in, 1, fixed_point_position_in);
    RawTensor ref_dst(shape, dt_out, 1, fixed_point_position_out);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::depth_convert(ref_src, ref_dst, policy, shift);

    return ref_dst;
}

RawTensor Reference::compute_reference_gaussian3x3(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::gaussian3x3(ref_src, ref_dst, border_mode, constant_border_value);

    return ref_dst;
}

RawTensor Reference::compute_reference_gaussian5x5(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::gaussian5x5(ref_src, ref_dst, border_mode, constant_border_value);

    return ref_dst;
}

RawTensor Reference::compute_reference_gemm(const TensorShape &src_shape1, const TensorShape &src_shape2, const TensorShape &src_shape3,
                                            const TensorShape &dst_shape, float alpha, float beta, DataType dt, int fixed_point_position)
{
    RawTensor src1(src_shape1, dt, 1, fixed_point_position);
    RawTensor src2(src_shape2, dt, 1, fixed_point_position);
    RawTensor src3(src_shape3, dt, 1, fixed_point_position);
    RawTensor dst(dst_shape, dt, 1, fixed_point_position);

    // Fill reference
    if(dt == DataType::F16 || dt == DataType::F32)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(src1, distribution, 0);
        library->fill(src2, distribution, 1);
        library->fill(src3, distribution, 2);
    }
    else
    {
        library->fill_tensor_uniform(src1, 0);
        library->fill_tensor_uniform(src2, 1);
        library->fill_tensor_uniform(src3, 2);
    }

    // Compute reference
    ReferenceCPP::gemm(src1, src2, src3, dst, alpha, beta);

    return dst;
}

RawTensor Reference::compute_reference_non_linear_filter(const TensorShape &shape, NonLinearFilterFunction function, unsigned int mask_size,
                                                         MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode, uint8_t constant_border_value)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::non_linear_filter(ref_src, ref_dst, function, mask_size, pattern, mask, border_mode, constant_border_value);

    return ref_dst;
}

RawTensor Reference::compute_reference_pixel_wise_multiplication(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, float scale, ConvertPolicy convert_policy,
                                                                 RoundingPolicy rounding_policy)
{
    // Create reference
    RawTensor ref_src1(shape, dt_in0);
    RawTensor ref_src2(shape, dt_in1);
    RawTensor ref_dst(shape, dt_out);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::pixel_wise_multiplication(ref_src1, ref_src2, ref_dst, scale, convert_policy, rounding_policy);

    return ref_dst;
}

RawTensor Reference::compute_reference_fixed_point_pixel_wise_multiplication(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, float scale, int fixed_point_position,
                                                                             ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    // Create reference
    RawTensor ref_src1(shape, dt_in0, 1, fixed_point_position);
    RawTensor ref_src2(shape, dt_in1, 1, fixed_point_position);
    RawTensor ref_dst(shape, dt_out, 1, fixed_point_position);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::fixed_point_pixel_wise_multiplication(ref_src1, ref_src2, ref_dst, scale, convert_policy, rounding_policy);

    return ref_dst;
}

template <typename T>
RawTensor Reference::compute_reference_table_lookup(const TensorShape &shape, DataType dt_inout, std::map<T, T> &lut)
{
    // Create reference
    RawTensor ref_src(shape, dt_inout);
    RawTensor ref_dst(shape, dt_inout);
    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::table_lookup(ref_src, ref_dst, lut);

    return ref_dst;
}
template RawTensor arm_compute::test::validation::Reference::compute_reference_table_lookup<uint8_t>(const TensorShape &shape, DataType dt_inout, std::map<uint8_t, uint8_t> &lut);
template RawTensor arm_compute::test::validation::Reference::compute_reference_table_lookup<int16_t>(const TensorShape &shape, DataType dt_inout, std::map<int16_t, int16_t> &lut);

RawTensor Reference::compute_reference_threshold(const TensorShape &shape, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
{
    // Create reference
    RawTensor ref_src(shape, DataType::U8);
    RawTensor ref_dst(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::threshold(ref_src, ref_dst, threshold, false_value, true_value, type, upper);

    return ref_dst;
}

RawTensor Reference::compute_reference_activation_layer(const TensorShape &shape, DataType dt, ActivationLayerInfo act_info, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src(shape, dt, 1, fixed_point_position);
    RawTensor ref_dst(shape, dt, 1, fixed_point_position);

    // Fill tensors
    switch(dt)
    {
        case DataType::QS8:
        {
            const std::pair<int8_t, int8_t> bounds = get_activation_layer_test_bounds<int8_t>(act_info.activation(), fixed_point_position);
            std::uniform_int_distribution<> distribution(bounds.first, bounds.second);
            library->fill(ref_src, distribution, 0);
            break;
        }
        case DataType::QS16:
        {
            const std::pair<int16_t, int16_t> bounds = get_activation_layer_test_bounds<int16_t>(act_info.activation(), fixed_point_position);
            std::uniform_int_distribution<> distribution(bounds.first, bounds.second);
            library->fill(ref_src, distribution, 0);
            break;
        }
        case DataType::F16:
        {
            const std::pair<half_float::half, half_float::half> bounds = get_activation_layer_test_bounds<half_float::half>(act_info.activation());
            std::uniform_real_distribution<> distribution(bounds.first, bounds.second);
            library->fill(ref_src, distribution, 0);
            break;
        }
        case DataType::F32:
        {
            const std::pair<float, float> bounds = get_activation_layer_test_bounds<float>(act_info.activation());
            std::uniform_real_distribution<> distribution(bounds.first, bounds.second);
            library->fill(ref_src, distribution, 0);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported");
            break;
        }
    }

    // Compute reference
    ReferenceCPP::activation_layer(ref_src, ref_dst, act_info);

    return ref_dst;
}

RawTensor Reference::compute_reference_batch_normalization_layer(const TensorShape &shape0, const TensorShape &shape1, DataType dt, float epsilon, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src(shape0, dt, 1, fixed_point_position);
    RawTensor ref_dst(shape0, dt, 1, fixed_point_position);
    RawTensor ref_mean(shape1, dt, 1, fixed_point_position);
    RawTensor ref_var(shape1, dt, 1, fixed_point_position);
    RawTensor ref_beta(shape1, dt, 1, fixed_point_position);
    RawTensor ref_gamma(shape1, dt, 1, fixed_point_position);

    // Fill tensors
    switch(dt)
    {
        case DataType::QS8:
        {
            const std::pair<int8_t, int8_t> bounds = get_batchnormalization_layer_test_bounds<int8_t>(fixed_point_position);
            std::uniform_int_distribution<> distribution(bounds.first, bounds.second);
            std::uniform_int_distribution<> distribution_var(0, bounds.second);
            fill_tensors(distribution, { 0, 1, 3, 4 }, &ref_src, &ref_mean, &ref_beta, &ref_gamma);
            fill_tensors(distribution_var, { 0 }, &ref_var);
            break;
        }
        case DataType::QS16:
        {
            const std::pair<int16_t, int16_t> bounds = get_batchnormalization_layer_test_bounds<int16_t>(fixed_point_position);
            std::uniform_int_distribution<> distribution(bounds.first, bounds.second);
            std::uniform_int_distribution<> distribution_var(0, bounds.second);
            fill_tensors(distribution, { 0, 1, 3, 4 }, &ref_src, &ref_mean, &ref_beta, &ref_gamma);
            fill_tensors(distribution_var, { 0 }, &ref_var);
            break;
        }
        case DataType::F16:
        {
            const std::pair<half_float::half, half_float::half> bounds = get_batchnormalization_layer_test_bounds<half_float::half>();
            std::uniform_real_distribution<> distribution(bounds.first, bounds.second);
            std::uniform_real_distribution<> distribution_var(0, bounds.second);
            fill_tensors(distribution, { 0, 1, 3, 4 }, &ref_src, &ref_mean, &ref_beta, &ref_gamma);
            fill_tensors(distribution_var, { 0 }, &ref_var);
            break;
        }
        case DataType::F32:
        {
            const std::pair<float, float> bounds = get_batchnormalization_layer_test_bounds<float>();
            std::uniform_real_distribution<> distribution(bounds.first, bounds.second);
            std::uniform_real_distribution<> distribution_var(0, bounds.second);
            fill_tensors(distribution, { 0, 1, 3, 4 }, &ref_src, &ref_mean, &ref_beta, &ref_gamma);
            fill_tensors(distribution_var, { 0 }, &ref_var);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported");
            break;
        }
    }

    // Compute reference
    ReferenceCPP::batch_normalization_layer(ref_src, ref_dst, ref_mean, ref_var, ref_beta, ref_gamma, epsilon, fixed_point_position);

    return ref_dst;
}

RawTensor Reference::compute_reference_convolution_layer(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, DataType dt,
                                                         const PadStrideInfo &conv_info, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src(input_shape, dt, 1, fixed_point_position);
    RawTensor ref_weights(weights_shape, dt, 1, fixed_point_position);
    RawTensor ref_bias(bias_shape, dt, 1, fixed_point_position);
    RawTensor ref_dst(output_shape, dt, 1, fixed_point_position);

    // Fill reference
    switch(dt)
    {
        case DataType::F32:
        case DataType::F16:
        {
            std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
            library->fill(ref_src, distribution, 0);
            library->fill(ref_weights, distribution, 1);
            library->fill(ref_bias, distribution, 2);
            break;
        }
        case DataType::QS16:
        case DataType::QS8:
        {
            library->fill_tensor_uniform(ref_src, 0);
            library->fill_tensor_uniform(ref_weights, 1);
            library->fill_tensor_uniform(ref_bias, 2);
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported");
            break;
        }
    }

    // Compute reference
    ReferenceCPP::convolution_layer(ref_src, ref_weights, ref_bias, ref_dst, conv_info);

    return ref_dst;
}

RawTensor Reference::compute_reference_depth_concatenate_layer(const std::vector<TensorShape> &shapes, DataType dt, int fixed_point_position)
{
    std::vector<std::unique_ptr<RawTensor>> ref_srcs{};
    TensorShape                             dst_shape = calculate_depth_concatenate_shape(shapes);

    // Create tensors
    for(const auto &shape : shapes)
    {
        ref_srcs.push_back(support::cpp14::make_unique<RawTensor>(shape, dt, 1, fixed_point_position));
    }
    RawTensor ref_dst(dst_shape, dt, 1, fixed_point_position);

    // Fill references
    for(unsigned int i = 0; i < ref_srcs.size(); ++i)
    {
        library->fill_tensor_uniform(*ref_srcs[i], i);
    }

    // Compute reference
    ReferenceCPP::depth_concatenate_layer(ref_srcs, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_fully_connected_layer(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape,
                                                             DataType dt, bool transpose_weights, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src(input_shape, dt, 1, fixed_point_position);
    RawTensor ref_bias(bias_shape, dt, 1, fixed_point_position);
    RawTensor ref_dst(output_shape, dt, 1, fixed_point_position);

    // Swap the first and second dimension of weights' shape if transpose_weights is true
    TensorShape ws = weights_shape;
    if(transpose_weights)
    {
        const size_t dimx = ws.x();
        ws.set(0, ws.y());
        ws.set(1, dimx);
    }

    RawTensor ref_weights(ws, dt, 1, fixed_point_position);

    // Fill reference
    if(dt == DataType::F16 || dt == DataType::F32)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(ref_src, distribution, 0);
        library->fill(ref_weights, distribution, 1);
        library->fill(ref_bias, distribution, 2);
    }
    else
    {
        library->fill_tensor_uniform(ref_src, 0);
        library->fill_tensor_uniform(ref_weights, 1);
        library->fill_tensor_uniform(ref_bias, 2);
    }

    // Compute reference
    ReferenceCPP::fully_connected_layer(ref_src, ref_weights, ref_bias, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_pooling_layer(const TensorShape &shape_in, const TensorShape &shape_out, DataType dt, PoolingLayerInfo pool_info, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src(shape_in, dt, 1, fixed_point_position);
    RawTensor ref_dst(shape_out, dt, 1, fixed_point_position);

    // Fill reference
    int min = 0;
    int max = 0;
    switch(dt)
    {
        case DataType::F32:
        case DataType::F16:
            min = -1;
            max = 1;
            break;
        case DataType::QS8:
            min = -(1 << fixed_point_position);
            max = (1 << fixed_point_position);
            break;
        default:
            ARM_COMPUTE_ERROR("DataType not supported.");
    }
    std::uniform_real_distribution<> distribution(min, max);
    library->fill(ref_src, distribution, 0.0);

    // Compute reference
    ReferenceCPP::pooling_layer(ref_src, ref_dst, pool_info, fixed_point_position);

    return ref_dst;
}

RawTensor Reference::compute_reference_roi_pooling_layer(const TensorShape &shape, DataType dt, const std::vector<ROI> &rois, const ROIPoolingLayerInfo &pool_info)
{
    TensorShape shape_dst;
    shape_dst.set(0, pool_info.pooled_width());
    shape_dst.set(1, pool_info.pooled_height());
    shape_dst.set(2, shape.z());
    shape_dst.set(3, rois.size());

    // Create reference
    RawTensor ref_src(shape, dt);
    RawTensor ref_dst(shape_dst, dt);

    // Fill reference
    std::uniform_real_distribution<> distribution(-1, 1);
    library->fill(ref_src, distribution, 0.0);

    // Compute reference
    ReferenceCPP::roi_pooling_layer(ref_src, ref_dst, rois, pool_info);

    return ref_dst;
}

RawTensor Reference::compute_reference_fixed_point_operation(const TensorShape &shape, DataType dt_in, DataType dt_out, FixedPointOp op, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src(shape, dt_in, 1, fixed_point_position);
    RawTensor ref_dst(shape, dt_out, 1, fixed_point_position);

    // Fill reference
    int min = 0;
    int max = 0;
    switch(op)
    {
        case(FixedPointOp::INV_SQRT):
            min = 1;
            max = (dt_in == DataType::QS8) ? 0x7F : 0x7FFF;
            break;
        case(FixedPointOp::LOG):
            min = (1 << (fixed_point_position - 1));
            max = (dt_in == DataType::QS8) ? 0x3F : 0x3FFF;
            break;
        case(FixedPointOp::EXP):
            min = -(1 << (fixed_point_position - 1));
            max = (1 << (fixed_point_position - 1));
            break;
        case(FixedPointOp::RECIPROCAL):
            min = 15;
            max = (dt_in == DataType::QS8) ? 0x7F : 0x7FFF;
            break;
        default:
            ARM_COMPUTE_ERROR("Fixed point operation not supported");
    }
    std::uniform_int_distribution<> distribution(min, max);
    library->fill(ref_src, distribution, 0);

    // Compute reference
    ReferenceCPP::fixed_point_operation(ref_src, ref_dst, op);

    return ref_dst;
}

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* DOXYGEN_SKIP_THIS */
