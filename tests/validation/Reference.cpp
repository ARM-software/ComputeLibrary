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

#include "Globals.h"
#include "Helpers.h"
#include "ReferenceCPP.h"
#include "TensorLibrary.h"
#include "validation/Helpers.h"

#include <random>

using namespace arm_compute::test;

namespace arm_compute
{
namespace test
{
namespace validation
{
std::pair<float, float> Reference::compute_reference_mean_and_standard_deviation(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src = library->get(shape, DataType::U8);

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
    RawTensor ref_src = library->get(shape, DataType::U8);
    RawTensor ref_dst = library->get(shape, DataType::U32);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::integral_image(ref_src, ref_dst);

    return ref_dst;
}
RawTensor Reference::compute_reference_absolute_difference(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out)
{
    // Create reference
    RawTensor ref_src1 = library->get(shape, dt_in0);
    RawTensor ref_src2 = library->get(shape, dt_in1);
    RawTensor ref_dst  = library->get(shape, dt_out);

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
    RawTensor ref_src = library->get(shape, DataType::U8);
    RawTensor ref_dst = library->get(shape, DataType::S16);

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
    RawTensor ref_src = library->get(shape, DataType::U8);
    RawTensor ref_dst = library->get(shape, DataType::S16);

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
    RawTensor ref_src = library->get(shape, DataType::U8);
    RawTensor ref_dst = library->get(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);
    library->fill_tensor_uniform(ref_dst, 1);

    // Compute reference
    ReferenceCPP::accumulate_weighted(ref_src, ref_dst, alpha);

    return ref_dst;
}

RawTensor Reference::compute_reference_arithmetic_addition(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, ConvertPolicy convert_policy)
{
    // Create reference
    RawTensor ref_src1 = library->get(shape, dt_in0);
    RawTensor ref_src2 = library->get(shape, dt_in1);
    RawTensor ref_dst  = library->get(shape, dt_out);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::arithmetic_addition(ref_src1, ref_src2, ref_dst, convert_policy);

    return ref_dst;
}

RawTensor Reference::compute_reference_arithmetic_subtraction(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, ConvertPolicy convert_policy)
{
    // Create reference
    RawTensor ref_src1 = library->get(shape, dt_in0);
    RawTensor ref_src2 = library->get(shape, dt_in1);
    RawTensor ref_dst  = library->get(shape, dt_out);

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
    RawTensor ref_src1 = library->get(shape, DataType::U8);
    RawTensor ref_src2 = library->get(shape, DataType::U8);
    RawTensor ref_dst  = library->get(shape, DataType::U8);

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
    RawTensor ref_src1 = library->get(shape, DataType::U8);
    RawTensor ref_src2 = library->get(shape, DataType::U8);
    RawTensor ref_dst  = library->get(shape, DataType::U8);

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
    RawTensor ref_src1 = library->get(shape, DataType::U8);
    RawTensor ref_src2 = library->get(shape, DataType::U8);
    RawTensor ref_dst  = library->get(shape, DataType::U8);

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
    RawTensor ref_src = library->get(shape, DataType::U8);
    RawTensor ref_dst = library->get(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::bitwise_not(ref_src, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_box3x3(const TensorShape &shape)
{
    // Create reference
    RawTensor ref_src = library->get(shape, DataType::U8);
    RawTensor ref_dst = library->get(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::box3x3(ref_src, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_depth_convert(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift, uint32_t fixed_point_position)
{
    RawTensor ref_src = library->get(shape, dt_in, 1, fixed_point_position);
    RawTensor ref_dst = library->get(shape, dt_out, 1, fixed_point_position);

    // Fill reference
    library->fill_tensor_uniform(ref_src, 0);

    // Compute reference
    ReferenceCPP::depth_convert(ref_src, ref_dst, policy, shift);

    return ref_dst;
}

RawTensor Reference::compute_reference_gemm(const TensorShape &src_shape1, const TensorShape &src_shape2, const TensorShape &src_shape3,
                                            const TensorShape &dst_shape, float alpha, float beta, DataType dt, int fixed_point_position)
{
    RawTensor src1 = library->get(src_shape1, dt, 1, fixed_point_position);
    RawTensor src2 = library->get(src_shape2, dt, 1, fixed_point_position);
    RawTensor src3 = library->get(src_shape3, dt, 1, fixed_point_position);
    RawTensor dst  = library->get(dst_shape, dt, 1, fixed_point_position);

    // Fill reference
    if(dt == DataType::F32)
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

RawTensor Reference::compute_reference_pixel_wise_multiplication(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, float scale, ConvertPolicy convert_policy,
                                                                 RoundingPolicy rounding_policy)
{
    // Create reference
    RawTensor ref_src1 = library->get(shape, dt_in0);
    RawTensor ref_src2 = library->get(shape, dt_in1);
    RawTensor ref_dst  = library->get(shape, dt_out);

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
    RawTensor ref_src1 = library->get(shape, dt_in0, 1, fixed_point_position);
    RawTensor ref_src2 = library->get(shape, dt_in1, 1, fixed_point_position);
    RawTensor ref_dst  = library->get(shape, dt_out, 1, fixed_point_position);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);
    library->fill_tensor_uniform(ref_src2, 1);

    // Compute reference
    ReferenceCPP::fixed_point_pixel_wise_multiplication(ref_src1, ref_src2, ref_dst, scale, convert_policy, rounding_policy);

    return ref_dst;
}

RawTensor Reference::compute_reference_threshold(const TensorShape &shape, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
{
    // Create reference
    RawTensor ref_src1 = library->get(shape, DataType::U8);
    RawTensor ref_dst  = library->get(shape, DataType::U8);

    // Fill reference
    library->fill_tensor_uniform(ref_src1, 0);

    // Compute reference
    ReferenceCPP::threshold(ref_src1, ref_dst, threshold, false_value, true_value, type, upper);

    return ref_dst;
}

RawTensor Reference::compute_reference_activation_layer(const TensorShape &shape, DataType dt, ActivationLayerInfo act_info, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src = library->get(shape, dt, 1, fixed_point_position);
    RawTensor ref_dst = library->get(shape, dt, 1, fixed_point_position);

    // Fill reference
    if(dt == DataType::F32)
    {
        float min_bound = 0;
        float max_bound = 0;
        std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<float>(act_info.activation());
        std::uniform_real_distribution<> distribution(min_bound, max_bound);
        library->fill(ref_src, distribution, 0);
    }
    else
    {
        int min_bound = 0;
        int max_bound = 0;
        std::tie(min_bound, max_bound) = get_activation_layer_test_bounds<int8_t>(act_info.activation(), fixed_point_position);
        std::uniform_int_distribution<> distribution(min_bound, max_bound);
        library->fill(ref_src, distribution, 0);
    }

    // Compute reference
    ReferenceCPP::activation_layer(ref_src, ref_dst, act_info);

    return ref_dst;
}

RawTensor Reference::compute_reference_batch_normalization_layer(const TensorShape &shape0, const TensorShape &shape1, DataType dt, float epsilon, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src   = library->get(shape0, dt, 1, fixed_point_position);
    RawTensor ref_dst   = library->get(shape0, dt, 1, fixed_point_position);
    RawTensor ref_mean  = library->get(shape1, dt, 1, fixed_point_position);
    RawTensor ref_var   = library->get(shape1, dt, 1, fixed_point_position);
    RawTensor ref_beta  = library->get(shape1, dt, 1, fixed_point_position);
    RawTensor ref_gamma = library->get(shape1, dt, 1, fixed_point_position);

    // Fill tensors with values from -1 to 1.
    if(dt == DataType::F32)
    {
        float min_bound = 0.f;
        float max_bound = 0.f;
        std::tie(min_bound, max_bound) = get_batchnormalization_layer_test_bounds<float>();
        std::uniform_real_distribution<> distribution(min_bound, max_bound);
        std::uniform_real_distribution<> distribution_var(0, max_bound);
        library->fill(ref_src, distribution, 0);
        library->fill(ref_mean, distribution, 1);
        library->fill(ref_var, distribution_var, 0);
        library->fill(ref_beta, distribution, 3);
        library->fill(ref_gamma, distribution, 4);
    }
    else
    {
        int min_bound = 0;
        int max_bound = 0;
        std::tie(min_bound, max_bound) = get_batchnormalization_layer_test_bounds<int8_t>(fixed_point_position);
        std::uniform_int_distribution<> distribution(min_bound, max_bound);
        std::uniform_int_distribution<> distribution_var(0, max_bound);
        library->fill(ref_src, distribution, 0);
        library->fill(ref_mean, distribution, 1);
        library->fill(ref_var, distribution_var, 0);
        library->fill(ref_beta, distribution, 3);
        library->fill(ref_gamma, distribution, 4);
    }

    // Compute reference
    ReferenceCPP::batch_normalization_layer(ref_src, ref_dst, ref_mean, ref_var, ref_beta, ref_gamma, epsilon, fixed_point_position);

    return ref_dst;
}

RawTensor Reference::compute_reference_convolution_layer(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, DataType dt,
                                                         const PadStrideInfo &conv_info, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src     = library->get(input_shape, dt, 1, fixed_point_position);
    RawTensor ref_weights = library->get(weights_shape, dt, 1, fixed_point_position);
    RawTensor ref_bias    = library->get(bias_shape, dt, 1, fixed_point_position);
    RawTensor ref_dst     = library->get(output_shape, dt, 1, fixed_point_position);

    // Fill reference
    if(dt == DataType::F32)
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
    ReferenceCPP::convolution_layer(ref_src, ref_weights, ref_bias, ref_dst, conv_info);

    return ref_dst;
}

RawTensor Reference::compute_reference_fully_connected_layer(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape,
                                                             DataType dt, bool transpose_weights, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src  = library->get(input_shape, dt, 1, fixed_point_position);
    RawTensor ref_bias = library->get(bias_shape, dt, 1, fixed_point_position);
    RawTensor ref_dst  = library->get(output_shape, dt, 1, fixed_point_position);

    // Swap the first and second dimension of weights' shape if transpose_weights is true
    TensorShape ws = weights_shape;
    if(transpose_weights)
    {
        const size_t dimx = ws.x();
        ws.set(0, ws.y());
        ws.set(1, dimx);
    }

    RawTensor ref_weights = library->get(ws, dt, 1, fixed_point_position);

    // Fill reference
    if(dt == DataType::F32)
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

RawTensor Reference::compute_reference_normalization_layer(const TensorShape &shape, DataType dt, NormalizationLayerInfo norm_info, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src = library->get(shape, dt, 1, fixed_point_position);
    RawTensor ref_dst = library->get(shape, dt, 1, fixed_point_position);

    // Fill reference
    if(dt == DataType::QS8)
    {
        const int8_t one_fixed_point       = 1 << fixed_point_position;
        const int8_t minus_one_fixed_point = -one_fixed_point;
        library->fill_tensor_uniform(ref_src, 0, minus_one_fixed_point, one_fixed_point);
    }
    else
    {
        library->fill_tensor_uniform(ref_src, 0);
    }

    // Compute reference
    ReferenceCPP::normalization_layer(ref_src, ref_dst, norm_info);

    return ref_dst;
}

RawTensor Reference::compute_reference_pooling_layer(const TensorShape &shape_in, const TensorShape &shape_out, DataType dt, PoolingLayerInfo pool_info, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src = library->get(shape_in, dt, 1, fixed_point_position);
    RawTensor ref_dst = library->get(shape_out, dt, 1, fixed_point_position);

    // Fill reference
    int min = 0;
    int max = 0;
    switch(dt)
    {
        case DataType::F32:
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

RawTensor Reference::compute_reference_softmax_layer(const TensorShape &shape, DataType dt, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src = library->get(shape, dt, 1, fixed_point_position);
    RawTensor ref_dst = library->get(shape, dt, 1, fixed_point_position);

    // Fill reference
    if(arm_compute::is_data_type_float(dt))
    {
        std::uniform_real_distribution<> distribution(-10, 10);
        library->fill(ref_src, distribution, 0);
    }
    else
    {
        int                             one_fixed = 1 << fixed_point_position;
        std::uniform_int_distribution<> distribution(-one_fixed, one_fixed);
        library->fill(ref_src, distribution, 0);
    }

    // Compute reference
    ReferenceCPP::softmax_layer(ref_src, ref_dst);

    return ref_dst;
}

RawTensor Reference::compute_reference_fixed_point_operation(const TensorShape &shape, DataType dt_in, DataType dt_out, FixedPointOp op, int fixed_point_position)
{
    // Create reference
    RawTensor ref_src = library->get(shape, dt_in, 1, fixed_point_position);
    RawTensor ref_dst = library->get(shape, dt_out, 1, fixed_point_position);

    // Fill reference
    int min = 0;
    int max = 0;
    switch(op)
    {
        case(FixedPointOp::INV_SQRT):
            min = 32;
            max = 127;
            break;
        case(FixedPointOp::LOG):
            min = (1 << (fixed_point_position - 1));
            max = 63;
            break;
        case(FixedPointOp::EXP):
            min = 1;
            max = (1 << (fixed_point_position - 1));
            break;
        case(FixedPointOp::RECIPROCAL):
            min = 15;
            max = 100;
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
