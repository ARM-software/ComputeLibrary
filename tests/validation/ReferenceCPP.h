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
#ifndef __ARM_COMPUTE_TEST_REFERENCE_REFERENCE_CPP_H__
#define __ARM_COMPUTE_TEST_REFERENCE_REFERENCE_CPP_H__

#include "Reference.h"

#include "RawTensor.h"

#include <ostream>

namespace arm_compute
{
class Tensor;

namespace test
{
namespace validation
{
/** C++ reference implementation. */
class ReferenceCPP final : public Reference
{
public:
    /** Function to compute the integral image of a tensor.
     *
     * @param[in]  src Input tensor.
     * @param[out] dst Result tensor.
     */
    static void integral_image(const RawTensor &src, RawTensor &dst);
    /** Function to compute the absolute difference between two tensors.
     *
     * @param[in]  src1 First tensor.
     * @param[in]  src2 Second tensor.
     * @param[out] dst  Result tensor.
     */
    static void absolute_difference(const RawTensor &src1, const RawTensor &src2, RawTensor &dst);
    /** Function to accumulate an input tensor into an output tensor.
     *
     * @param[in]      src Input tensor.
     * @param[in, out] dst Result tensor.
     */
    static void accumulate(const RawTensor &src, RawTensor &dst);
    /** Function to accumulate a squared value from an input tensor to an output tensor.
     *
     * @param[in]      src   Input tensor.
     * @param[in, out] dst   Result tensor.
     * @param[in]      shift A uint32_t value within the range of [0, 15]
     */
    static void accumulate_squared(const RawTensor &src, RawTensor &dst, uint32_t shift);
    /** Function to accumulate a weighted value from an input tensor to an output tensor.
     *
     * @param[in]      src   Input tensor.
     * @param[in, out] dst   Result tensor.
     * @param[in]      alpha A float value within the range of [0, 1]
     */
    static void accumulate_weighted(const RawTensor &src, RawTensor &dst, float alpha);
    /** Arithmetic addition of @p src1 and @p src2
     *
     * @param[in]  src1           First tensor.
     * @param[in]  src2           Second tensor.
     * @param[out] dst            Result tensor.
     * @param[in]  convert_policy Overflow policy.
     */
    static void arithmetic_addition(const RawTensor &src1, const RawTensor &src2, RawTensor &dst, ConvertPolicy convert_policy);
    /** Arithmetic subtraction of @p src2 from @p src1
     *
     * @param[in]  src1           First tensor.
     * @param[in]  src2           Second tensor.
     * @param[out] dst            Result tensor.
     * @param[in]  convert_policy Overflow policy.
     */
    static void arithmetic_subtraction(const RawTensor &src1, const RawTensor &src2, RawTensor &dst, ConvertPolicy convert_policy);
    /** Function to compute the bitwise and between two tensors.
     *
     * @param[in]  src1 First tensor.
     * @param[in]  src2 Second tensor.
     * @param[out] dst  Result tensor.
     */
    static void bitwise_and(const RawTensor &src1, const RawTensor &src2, RawTensor &dst);
    /** Function to compute the bitwise or between two tensors.
     *
     * @param[in]  src1 First tensor.
     * @param[in]  src2 Second tensor.
     * @param[out] dst  Result tensor.
     */
    static void bitwise_or(const RawTensor &src1, const RawTensor &src2, RawTensor &dst);
    /** Function to compute the bitwise xor between two tensors.
     *
     * @param[in]  src1 First tensor.
     * @param[in]  src2 Second tensor.
     * @param[out] dst  Result tensor.
     */
    static void bitwise_xor(const RawTensor &src1, const RawTensor &src2, RawTensor &dst);
    /** Function to compute the bitwise not of a tensor.
     *
     * @param[in]  src Input tensor.
     * @param[out] dst Result tensor.
     */
    static void bitwise_not(const RawTensor &src, RawTensor &dst);
    /** Function to compute 3-by-3 box filtered result tensor.
     *
     * @param[in]  src Input tensor.
     * @param[out] dst Result tensor.
     */
    static void box3x3(const RawTensor &src, RawTensor &dst);
    /** Depth conversion from @p src to @p dst
     *
     * @param[in]  src    First tensor.
     * @param[out] dst    Result tensor.
     * @param[in]  policy Overflow policy.
     * @param[in]  shift  Value for down/up conversions.
     */
    static void depth_convert(const RawTensor &src, RawTensor &dst, ConvertPolicy policy, uint32_t shift);
    /** Compute GEMM function.
     *
     * @param[in]  src1  First input tensor
     * @param[in]  src2  Second input tensor
     * @param[in]  src3  Third input tensor
     * @param[out] dst   Output tensr
     * @param[in]  alpha Weight of the matrix product
     * @param[in]  beta  Weight of the third matrix
     */
    static void gemm(const RawTensor &src1, const RawTensor &src2, const RawTensor &src3,
                     RawTensor &dst, float alpha, float beta);
    /** Element-wise multiplication of @p src1, @p src2 and @p scale
     *
     * @param[in]  src1            First tensor.
     * @param[in]  src2            Second tensor.
     * @param[out] dst             Result tensor.
     * @param[in]  scale           A non-negative float multiplied to each product.
     * @param[in]  convert_policy  Overflow policy.
     * @param[in]  rounding_policy Rounding policy.
     */
    static void pixel_wise_multiplication(const RawTensor &src1, const RawTensor &src2, RawTensor &dst, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy);
    /** Fixed-point Pixel-wise multiplication of @p src1 by @p src2
     *
     * @param[in]  src1            First tensor.
     * @param[in]  src2            Second tensor.
     * @param[out] dst             Result tensor.
     * @param[in]  scale           A non-negative float multiplied to each product.
     * @param[in]  convert_policy  Overflow policy.
     * @param[in]  rounding_policy Rounding policy.
     */
    static void fixed_point_pixel_wise_multiplication(const RawTensor &src1, const RawTensor &src2, RawTensor &dst, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy);
    /** Threshold of@p src to @p dst
     *
     * @param[in]  src         First tensor.
     * @param[out] dst         Result tensor.
     * @param[in]  threshold   Threshold. When the threhold type is RANGE, this is used as the lower threshold.
     * @param[in]  false_value value to set when the condition is not respected.
     * @param[in]  true_value  value to set when the condition is respected.
     * @param[in]  type        Thresholding type. Either RANGE or BINARY.
     * @param[in]  upper       Upper threshold. Only used when the thresholding type is RANGE.
     */
    static void threshold(const RawTensor &src, RawTensor &dst, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper);
    /** Activation layer of @p src base on information from @p act_info.
     *
     * @param[in]  input    Input tensor.
     * @param[in]  output   Second tensor.
     * @param[out] act_info Activation layer information.
     */
    static void activation_layer(const RawTensor &input, RawTensor &output, ActivationLayerInfo act_info);
    /** Batch Normalization of @p src based on the information from @p norm_info.
     *
     * @param[in]  src                  Input tensor.
     * @param[out] dst                  Result tensor.
     * @param[out] mean                 Mean vector tensor.
     * @param[out] var                  Var vector tensor.
     * @param[out] beta                 Beta vector tensor.
     * @param[out] gamma                Gamma vector tensor.
     * @param[in]  epsilon              Small value to avoid division with zero.
     * @param[in]  fixed_point_position Fixed point position.
     */
    static void batch_normalization_layer(const RawTensor &src, RawTensor &dst, const RawTensor &mean, const RawTensor &var, const RawTensor &beta, const RawTensor &gamma, float epsilon,
                                          int fixed_point_position = 0);
    /** Convolution layer function
     *
     * @param[in]  src       Input tensor.
     * @param[in]  weights   Weights tensor.
     * @param[in]  bias      Bias tensor.
     * @param[out] dst       Result tensor.
     * @param[in]  conv_info Pads and strides information for the convolution layer.
     */
    static void convolution_layer(const RawTensor &src, const RawTensor &weights, const RawTensor &bias, RawTensor &dst, const PadStrideInfo &conv_info);
    /** Fully connected layer function
     *
     * @param[in]  src     Input tensor
     * @param[in]  weights Weights tensor.
     * @param[in]  bias    Bias tensor.
     * @param[out] dst     Result tensor.
     */
    static void fully_connected_layer(const RawTensor &src, const RawTensor &weights, const RawTensor &bias, RawTensor &dst);
    /** Normalization of @p src based on the information from @p norm_info.
     *
     * @param[in]  src       Input tensor.
     * @param[out] dst       Result tensor.
     * @param[in]  norm_info Normalization Layer information.
     */
    static void normalization_layer(const RawTensor &src, RawTensor &dst, NormalizationLayerInfo norm_info);
    /** Pooling layer of @p src based on the information from @p norm_info.
     *
     * @param[in]  src                  Input tensor.
     * @param[out] dst                  Result tensor.
     * @param[in]  pool_info            Pooling Layer information.
     * @param[in]  fixed_point_position Fixed point position. (Optional)
     */
    static void pooling_layer(const RawTensor &src, RawTensor &dst, PoolingLayerInfo pool_info, int fixed_point_position = 0);
    /** Softmax Layer of @p src.
     *
     * @param[in]  src Input tensor.
     * @param[out] dst Result tensor.
     */
    static void softmax_layer(const RawTensor &src, RawTensor &dst);
    /** Fixed point operations of @p src
     *
     * @param[in]  src Input tensor.
     * @param[out] dst Result tensor.
     * @param[in]  op  Fixed point operation to perform.
     */
    static void fixed_point_operation(const RawTensor &src, RawTensor &dst, FixedPointOp op);

private:
    ReferenceCPP()  = delete;
    ~ReferenceCPP() = delete;
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif
