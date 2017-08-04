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
#include "ReferenceCPP.h"

#include "TensorFactory.h"
#include "TensorOperations.h"
#include "TensorVisitors.h"
#include "TypePrinter.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/runtime/Tensor.h"

#include "boost_wrapper.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

using namespace arm_compute::test::validation::tensor_visitors;

namespace arm_compute
{
namespace test
{
namespace validation
{
// Sobel 3x3
void ReferenceCPP::sobel_3x3(RawTensor &src, RawTensor &dst_x, RawTensor &dst_y, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst_x.data_type() != DataType::S16 || dst_y.data_type() != DataType::S16);
    Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<int16_t> dx(dst_x.shape(), dst_x.data_type(), dst_x.fixed_point_position(), reinterpret_cast<int16_t *>(dst_x.data()));
    Tensor<int16_t> dy(dst_y.shape(), dst_y.data_type(), dst_y.fixed_point_position(), reinterpret_cast<int16_t *>(dst_y.data()));
    tensor_operations::sobel_3x3(s, dx, dy, border_mode, constant_border_value);
}

// Sobel 5x5
void ReferenceCPP::sobel_5x5(RawTensor &src, RawTensor &dst_x, RawTensor &dst_y, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst_x.data_type() != DataType::S16 || dst_y.data_type() != DataType::S16);
    Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<int16_t> dx(dst_x.shape(), dst_x.data_type(), dst_x.fixed_point_position(), reinterpret_cast<int16_t *>(dst_x.data()));
    Tensor<int16_t> dy(dst_y.shape(), dst_y.data_type(), dst_y.fixed_point_position(), reinterpret_cast<int16_t *>(dst_y.data()));
    tensor_operations::sobel_5x5(s, dx, dy, border_mode, constant_border_value);
}

// Minimum maximum location
void ReferenceCPP::min_max_location(const RawTensor &src, void *min, void *max, IArray<Coordinates2D> &min_loc, IArray<Coordinates2D> &max_loc, uint32_t &min_count, uint32_t &max_count)
{
    const TensorVariant s = TensorFactory::get_tensor(src);
    boost::apply_visitor(tensor_visitors::min_max_location_visitor(min, max, min_loc, max_loc, min_count, max_count), s);
}

// Absolute difference
void ReferenceCPP::absolute_difference(const RawTensor &src1, const RawTensor &src2, RawTensor &dst)
{
    const TensorVariant s1 = TensorFactory::get_tensor(src1);
    const TensorVariant s2 = TensorFactory::get_tensor(src2);
    TensorVariant       d  = TensorFactory::get_tensor(dst);
    boost::apply_visitor(absolute_difference_visitor(), s1, s2, d);
}

// Mean and standard deviation
void ReferenceCPP::mean_and_standard_deviation(const RawTensor &src, float &mean, float &std_dev)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    tensor_operations::mean_and_standard_deviation(s, mean, std_dev);
}

// Integral image
void ReferenceCPP::integral_image(const RawTensor &src, RawTensor &dst)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::U32);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<uint32_t>      d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint32_t *>(dst.data()));
    tensor_operations::integral_image(s, d);
}

// Accumulate
void ReferenceCPP::accumulate(const RawTensor &src, RawTensor &dst)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::S16);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<int16_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<int16_t *>(dst.data()));
    tensor_operations::accumulate(s, d);
}

// Accumulate squared
void ReferenceCPP::accumulate_squared(const RawTensor &src, RawTensor &dst, uint32_t shift)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::S16);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<int16_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<int16_t *>(dst.data()));
    tensor_operations::accumulate_squared(s, d, shift);
}

// Accumulate weighted
void ReferenceCPP::accumulate_weighted(const RawTensor &src, RawTensor &dst, float alpha)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::U8);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<uint8_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint8_t *>(dst.data()));
    tensor_operations::accumulate_weighted(s, d, alpha);
}

// Arithmetic addition
void ReferenceCPP::arithmetic_addition(const RawTensor &src1, const RawTensor &src2, RawTensor &dst, ConvertPolicy convert_policy)
{
    const TensorVariant s1 = TensorFactory::get_tensor(src1);
    const TensorVariant s2 = TensorFactory::get_tensor(src2);
    TensorVariant       d  = TensorFactory::get_tensor(dst);
    boost::apply_visitor(arithmetic_addition_visitor(convert_policy), s1, s2, d);
}

// Arithmetic subtraction
void ReferenceCPP::arithmetic_subtraction(const RawTensor &src1, const RawTensor &src2, RawTensor &dst, ConvertPolicy convert_policy)
{
    const TensorVariant s1 = TensorFactory::get_tensor(src1);
    const TensorVariant s2 = TensorFactory::get_tensor(src2);
    TensorVariant       d  = TensorFactory::get_tensor(dst);
    boost::apply_visitor(arithmetic_subtraction_visitor(convert_policy), s1, s2, d);
}

// Bitwise xor
void ReferenceCPP::bitwise_xor(const RawTensor &src1, const RawTensor &src2, RawTensor &dst)
{
    ARM_COMPUTE_ERROR_ON(src1.data_type() != DataType::U8 || src2.data_type() != DataType::U8 || dst.data_type() != DataType::U8);
    const Tensor<uint8_t> s1(src1.shape(), src1.data_type(), src1.fixed_point_position(), reinterpret_cast<const uint8_t *>(src1.data()));
    const Tensor<uint8_t> s2(src2.shape(), src2.data_type(), src2.fixed_point_position(), reinterpret_cast<const uint8_t *>(src2.data()));
    Tensor<uint8_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint8_t *>(dst.data()));
    tensor_operations::bitwise_xor(s1, s2, d);
}

// Bitwise not
void ReferenceCPP::bitwise_not(const RawTensor &src, RawTensor &dst)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::U8);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<uint8_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint8_t *>(dst.data()));
    tensor_operations::bitwise_not(s, d);
}

// Box3x3 filter
void ReferenceCPP::box3x3(const RawTensor &src, RawTensor &dst, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::U8);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<uint8_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint8_t *>(dst.data()));
    tensor_operations::box3x3(s, d, border_mode, constant_border_value);
}

// Depth conversion
void ReferenceCPP::depth_convert(const RawTensor &src, RawTensor &dst, ConvertPolicy policy, uint32_t shift)
{
    const TensorVariant s = TensorFactory::get_tensor(src);
    TensorVariant       d = TensorFactory::get_tensor(dst);
    boost::apply_visitor(tensor_visitors::depth_convert_visitor(policy, shift), s, d);
}

// Gaussian3x3 filter
void ReferenceCPP::gaussian3x3(const RawTensor &src, RawTensor &dst, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::U8);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<uint8_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint8_t *>(dst.data()));
    tensor_operations::gaussian3x3(s, d, border_mode, constant_border_value);
}

// Gaussian5x5 filter
void ReferenceCPP::gaussian5x5(const RawTensor &src, RawTensor &dst, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::U8);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<uint8_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint8_t *>(dst.data()));
    tensor_operations::gaussian5x5(s, d, border_mode, constant_border_value);
}

// Non linear filter
void ReferenceCPP::non_linear_filter(const RawTensor &src, RawTensor &dst, NonLinearFilterFunction function, unsigned int mask_size,
                                     MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::U8);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<uint8_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint8_t *>(dst.data()));
    tensor_operations::non_linear_filter(s, d, function, mask_size, pattern, mask, border_mode, constant_border_value);
}

// Pixel-wise multiplication
void ReferenceCPP::pixel_wise_multiplication(const RawTensor &src1, const RawTensor &src2, RawTensor &dst, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    const TensorVariant s1 = TensorFactory::get_tensor(src1);
    const TensorVariant s2 = TensorFactory::get_tensor(src2);
    TensorVariant       d  = TensorFactory::get_tensor(dst);
    boost::apply_visitor(pixel_wise_multiplication_visitor(scale, convert_policy, rounding_policy), s1, s2, d);
}

// Fixed-point Pixel-wise multiplication
void ReferenceCPP::fixed_point_pixel_wise_multiplication(const RawTensor &src1, const RawTensor &src2, RawTensor &dst, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
{
    const TensorVariant s1 = TensorFactory::get_tensor(src1);
    const TensorVariant s2 = TensorFactory::get_tensor(src2);
    TensorVariant       d  = TensorFactory::get_tensor(dst);
    boost::apply_visitor(tensor_visitors::fixed_point_pixel_wise_multiplication_visitor(s1, s2, scale, convert_policy, rounding_policy), d);
}

// Table lookup
template <typename T>
void ReferenceCPP::table_lookup(const RawTensor &src, RawTensor &dst, std::map<T, T> &lut)
{
    const TensorVariant s = TensorFactory::get_tensor(src);
    TensorVariant       d = TensorFactory::get_tensor(dst);
    boost::apply_visitor(tensor_visitors::table_lookup<T>(s, lut), d);
}
#ifndef DOXYGEN_SKIP_THIS
template void arm_compute::test::validation::ReferenceCPP::table_lookup<uint8_t>(const RawTensor &src, RawTensor &dst, std::map<uint8_t, uint8_t> &lut);
template void arm_compute::test::validation::ReferenceCPP::table_lookup<int16_t>(const RawTensor &src, RawTensor &dst, std::map<int16_t, int16_t> &lut);
#endif /* DOXYGEN_SKIP_THIS */

// Threshold
void ReferenceCPP::threshold(const RawTensor &src, RawTensor &dst, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
{
    ARM_COMPUTE_ERROR_ON(src.data_type() != DataType::U8 || dst.data_type() != DataType::U8);
    const Tensor<uint8_t> s(src.shape(), src.data_type(), src.fixed_point_position(), reinterpret_cast<const uint8_t *>(src.data()));
    Tensor<uint8_t>       d(dst.shape(), dst.data_type(), dst.fixed_point_position(), reinterpret_cast<uint8_t *>(dst.data()));
    tensor_operations::threshold(s, d, threshold, false_value, true_value, type, upper);
}

// Batch Normalization Layer
void ReferenceCPP::batch_normalization_layer(const RawTensor &src, RawTensor &dst, const RawTensor &mean, const RawTensor &var, const RawTensor &beta, const RawTensor &gamma, float epsilon,
                                             int fixed_point_position)
{
    const TensorVariant s = TensorFactory::get_tensor(src);
    TensorVariant       d = TensorFactory::get_tensor(dst);
    const TensorVariant m = TensorFactory::get_tensor(mean);
    const TensorVariant v = TensorFactory::get_tensor(var);
    const TensorVariant b = TensorFactory::get_tensor(beta);
    const TensorVariant g = TensorFactory::get_tensor(gamma);
    boost::apply_visitor(tensor_visitors::batch_normalization_layer_visitor(s, m, v, b, g, epsilon, fixed_point_position), d);
}

// Fully connected layer
void ReferenceCPP::fully_connected_layer(const RawTensor &src, const RawTensor &weights, const RawTensor &bias, RawTensor &dst)
{
    const TensorVariant s = TensorFactory::get_tensor(src);
    const TensorVariant w = TensorFactory::get_tensor(weights);
    const TensorVariant b = TensorFactory::get_tensor(bias);
    TensorVariant       d = TensorFactory::get_tensor(dst);
    boost::apply_visitor(tensor_visitors::fully_connected_layer_visitor(s, w, b), d);
}

// Pooling Layer
void ReferenceCPP::pooling_layer(const RawTensor &src, RawTensor &dst, PoolingLayerInfo pool_info, int fixed_point_position)
{
    const TensorVariant s = TensorFactory::get_tensor(src);
    TensorVariant       d = TensorFactory::get_tensor(dst);
    boost::apply_visitor(tensor_visitors::pooling_layer_visitor(s, pool_info, fixed_point_position), d);
}

// ROI Pooling Layer
void ReferenceCPP::roi_pooling_layer(const RawTensor &src, RawTensor &dst, const std::vector<ROI> &rois, const ROIPoolingLayerInfo &pool_info)
{
    const TensorVariant s = TensorFactory::get_tensor(src);
    TensorVariant       d = TensorFactory::get_tensor(dst);
    boost::apply_visitor(tensor_visitors::roi_pooling_layer_visitor(s, rois, pool_info), d);
}

// Fixed point operation
void ReferenceCPP::fixed_point_operation(const RawTensor &src, RawTensor &dst, FixedPointOp op)
{
    const TensorVariant s = TensorFactory::get_tensor(src);
    TensorVariant       d = TensorFactory::get_tensor(dst);
    boost::apply_visitor(tensor_visitors::fixed_point_operation_visitor(s, op), d);
}

} // namespace validation
} // namespace test
} // namespace arm_compute
