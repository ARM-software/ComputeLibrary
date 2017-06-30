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
#ifndef __ARM_COMPUTE_TEST_REFERENCE_REFERENCE_H__
#define __ARM_COMPUTE_TEST_REFERENCE_REFERENCE_H__

#include "RawTensor.h"
#include "Types.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Interface for reference implementations. */
class Reference
{
public:
    /** Compute reference sobel 3x3.
     *
     * @param[in] shape                 Shape of the input and output tensors.
     * @param[in] border_mode           Border mode to use for input tensor
     * @param[in] constant_border_value Constant value to use if @p border_mode is constant
     *
     * @return Computed raw tensors along x and y axis.
     */
    static std::pair<RawTensor, RawTensor> compute_reference_sobel_3x3(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value);
    /** Compute reference sobel 5x5.
     *
     * @param[in] shape                 Shape of the input and output tensors.
     * @param[in] border_mode           Border mode to use for input tensor
     * @param[in] constant_border_value Constant value to use if @p border_mode is constant
     *
     * @return Computed raw tensors along x and y axis.
     */
    static std::pair<RawTensor, RawTensor> compute_reference_sobel_5x5(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value);
    /** Compute reference mean and standard deviation.
     *
     * @param[in] shape Shape of the input tensors.
     *
     * @return Computed mean and standard deviation.
     */
    static std::pair<float, float> compute_reference_mean_and_standard_deviation(const TensorShape &shape);
    /** Compute reference integral image.
     *
     * @param[in] shape Shape of the input and output tensors.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_integral_image(const TensorShape &shape);
    /** Compute reference absolute difference.
     *
     * @param[in] shape  Shape of the input and output tensors.
     * @param[in] dt_in0 Data type of first input tensor.
     * @param[in] dt_in1 Data type of second input tensor.
     * @param[in] dt_out Data type of the output tensor.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_absolute_difference(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out);
    /** Compute reference accumulate.
     *
     * @param[in] shape Shape of the input and output tensors.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_accumulate(const TensorShape &shape);
    /** Compute reference accumulate.
     *
     * @param[in] shape Shape of the input and output tensors.
     * @param[in] shift A uint32_t value within the range of [0, 15]
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_accumulate_squared(const TensorShape &shape, uint32_t shift);
    /** Compute reference accumulate.
     *
     * @param[in] shape Shape of the input and output tensors.
     * @param[in] alpha A float value within the range of [0, 1]
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_accumulate_weighted(const TensorShape &shape, float alpha);
    /** Compute reference arithmetic addition.
     *
     * @param[in] shape          Shape of the input and output tensors.
     * @param[in] dt_in0         Data type of first input tensor.
     * @param[in] dt_in1         Data type of second input tensor.
     * @param[in] dt_out         Data type of the output tensor.
     * @param[in] convert_policy Overflow policy of the operation.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_arithmetic_addition(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, ConvertPolicy convert_policy);
    /** Compute reference arithmetic subtraction.
     *
     * @param[in] shape          Shape of the input and output tensors.
     * @param[in] dt_in0         Data type of first input tensor.
     * @param[in] dt_in1         Data type of second input tensor.
     * @param[in] dt_out         Data type of the output tensor.
     * @param[in] convert_policy Overflow policy of the operation.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_arithmetic_subtraction(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, ConvertPolicy convert_policy);
    /** Compute reference bitwise and.
     *
     * @param[in] shape Shape of the input and output tensors.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_bitwise_and(const TensorShape &shape);
    /** Compute reference bitwise or.
     *
     * @param[in] shape Shape of the input and output tensors.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_bitwise_or(const TensorShape &shape);
    /** Compute reference bitwise xor.
     *
     * @param[in] shape Shape of the input and output tensors.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_bitwise_xor(const TensorShape &shape);
    /** Compute reference bitwise not.
     *
     * @param[in] shape Shape of the input and output tensors.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_bitwise_not(const TensorShape &shape);
    /** Compute reference box3x3 filter.
     *
     * @param[in] shape                 Shape of the input and output tensors.
     * @param[in] border_mode           BorderMode used by the input tensor.
     * @param[in] constant_border_value Constant to use if @p border_mode == CONSTANT.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_box3x3(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value);
    /** Compute reference depth convert.
     *
     * @param[in] shape                Shape of the input and output tensors.
     * @param[in] dt_in                Data type of input tensor.
     * @param[in] dt_out               Data type of the output tensor.
     * @param[in] policy               Overflow policy of the operation.
     * @param[in] shift                Value for down/up conversions. Must be 0 <= shift < 8.
     * @param[in] fixed_point_position Fixed point position.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_depth_convert(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift, uint32_t fixed_point_position);
    /** Compute reference gaussian3x3 filter.
     *
     * @param[in] shape                 Shape of the input and output tensors.
     * @param[in] border_mode           BorderMode used by the input tensor
     * @param[in] constant_border_value Constant to use if @p border_mode == CONSTANT
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_gaussian3x3(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value);
    /** Compute reference gaussian5x5 filter.
     *
     * @param[in] shape                 Shape of the input and output tensors.
     * @param[in] border_mode           BorderMode used by the input tensor.
     * @param[in] constant_border_value Constant to use if @p border_mode == CONSTANT.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_gaussian5x5(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value);
    /** Compute matrix multiply function.
     *
     * @param[in]  src_shape1           First input tensor shape
     * @param[in]  src_shape2           Second input tensor shape
     * @param[in]  src_shape3           Third input tensor shape
     * @param[out] dst_shape            Output tensor.
     * @param[in]  alpha                Weight of the matrix product
     * @param[in]  beta                 Weight of the third matrix
     * @param[in]  dt                   Tensor's data type
     * @param[in]  fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers
     *
     * @return Computed output tensor.
     */
    static RawTensor compute_reference_gemm(const TensorShape &src_shape1, const TensorShape &src_shape2, const TensorShape &src_shape3,
                                            const TensorShape &dst_shape, float alpha, float beta, DataType dt, int fixed_point_position = 0);
    /** Compute reference non linear filter function
     *
     * @param[in] shape                 Shape of the input and output tensors.Data type supported: U8
     * @param[in] function              Non linear function to perform
     * @param[in] mask_size             Mask size. Supported sizes: 3, 5
     * @param[in] pattern               Matrix pattern
     * @param[in] mask                  The given mask. Will be used only if pattern is specified to PATTERN_OTHER
     * @param[in] border_mode           Strategy to use for borders.
     * @param[in] constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     *
     * @return Computed raw tensor.
    */
    static RawTensor compute_reference_non_linear_filter(const TensorShape &shape, NonLinearFilterFunction function, unsigned int mask_size,
                                                         MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode, uint8_t constant_border_value = 0);
    /** Compute reference pixel-wise multiplication
     *
     * @param[in] shape           Shape of the input and output tensors.
     * @param[in] dt_in0          Data type of first input tensor.
     * @param[in] dt_in1          Data type of second input tensor.
     * @param[in] dt_out          Data type of the output tensor.
     * @param[in] scale           Non-negative scale.
     * @param[in] convert_policy  Overflow policy of the operation.
     * @param[in] rounding_policy Rounding policy of the operation.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_pixel_wise_multiplication(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, float scale, ConvertPolicy convert_policy,
                                                                 RoundingPolicy rounding_policy);
    /** Compute reference pixel-wise multiplication.
     *
     * @param[in] shape                Shape of the input and output tensors.
     * @param[in] dt_in0               Data type of first input tensor.
     * @param[in] dt_in1               Data type of second input tensor.
     * @param[in] dt_out               Data type of the output tensor.
     * @param[in] scale                Scale to apply after multiplication. Must be positive.
     * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number.
     * @param[in] convert_policy       Overflow policy of the operation.
     * @param[in] rounding_policy      Rounding policy of the operation.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_fixed_point_pixel_wise_multiplication(const TensorShape &shape, DataType dt_in0, DataType dt_in1, DataType dt_out, float scale, int fixed_point_position,
                                                                             ConvertPolicy convert_policy, RoundingPolicy rounding_policy);
    /** Compute reference threshold.
     *
     * @param[in] shape       Shape of the input and output tensors.
     * @param[in] threshold   Threshold. When the threshold type is RANGE, this is used as the lower threshold.
     * @param[in] false_value value to set when the condition is not respected.
     * @param[in] true_value  value to set when the condition is respected.
     * @param[in] type        Thresholding type. Either RANGE or BINARY.
     * @param[in] upper       Upper threshold. Only used when the thresholding type is RANGE.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_threshold(const TensorShape &shape, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper);
    /** Compute reference activation layer.
     *
     * @param[in] shape                Shape of the input and output tensors.
     * @param[in] dt                   Data type of the tensors.
     * @param[in] act_info             Activation layer information.
     * @param[in] fixed_point_position (Optional)Number of bits for the fractional part of fixed point numbers.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_activation_layer(const TensorShape &shape, DataType dt, ActivationLayerInfo act_info, int fixed_point_position = 0);
    /** Compute reference batch normalization layer.
     *
     * @param[in] shape0               Shape of the input and output tensors.
     * @param[in] shape1               Shape of the vector tensors.
     * @param[in] dt                   Data type of all input and output tensors.
     * @param[in] epsilon              Small value to avoid division with zero.
     * @param[in] fixed_point_position Fixed point position.
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_batch_normalization_layer(const TensorShape &shape0, const TensorShape &shape1, DataType dt, float epsilon, int fixed_point_position = 0);
    /** Compute reference pixel-wise multiplication
     *
     * @param[in] input_shape          Shape for the input tensor
     * @param[in] weights_shape        Shape for the weights tensor
     * @param[in] bias_shape           Shape for the bias tensor
     * @param[in] output_shape         Shape for the output tensor
     * @param[in] dt                   Data type to use
     * @param[in] conv_info            Pads and strides information for the convolution layer
     * @param[in] fixed_point_position Number of bits for the fractional part of the fixed point numbers
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_convolution_layer(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, DataType dt,
                                                         const PadStrideInfo &conv_info, int fixed_point_position);
    /** Compute reference for fully connected layer function
     *
     * @param[in] input_shape          Shape for the input tensor
     * @param[in] weights_shape        Shape for the weights tensor
     * @param[in] bias_shape           Shape for the bias tensor
     * @param[in] output_shape         Shape for the output tensor
     * @param[in] dt                   Data type to use
     * @param[in] transpose_weights    Transpose the weights if true
     * @param[in] fixed_point_position Number of bits for the fractional part of the fixed point numbers
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_fully_connected_layer(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, DataType dt,
                                                             bool transpose_weights, int fixed_point_position);
    /** Compute reference normalization layer.
     *
     * @param[in] shape                Shape of the input and output tensors.
     * @param[in] dt                   Data type of input and output tensors.
     * @param[in] norm_info            Normalization Layer information.
     * @param[in] fixed_point_position (Optional) Fixed point position that expresses the number of bits for the fractional part of the number when the tensor's data type is QS8 or QS16 (default = 0).
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_normalization_layer(const TensorShape &shape, DataType dt, NormalizationLayerInfo norm_info, int fixed_point_position = 0);
    /** Compute reference pooling layer.
      *
      * @param[in] shape_in             Shape of the input tensor.
      * @param[in] shape_out            Shape of the output tensor.
      * @param[in] dt                   Data type of input and output tensors.
      * @param[in] pool_info            Pooling Layer information.
      * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers.
      *
      * @return Computed raw tensor.
      */
    static RawTensor compute_reference_pooling_layer(const TensorShape &shape_in, const TensorShape &shape_out, DataType dt, PoolingLayerInfo pool_info, int fixed_point_position = 0);
    /** Compute reference roi pooling layer.
     *
     * @param[in] shape     Shape of the input tensor.
     * @param[in] dt        Data type of input and output tensors.
     * @param[in] rois      Region of interest vector.
     * @param[in] pool_info ROI Pooling Layer information.
     */
    static RawTensor compute_reference_roi_pooling_layer(const TensorShape &shape, DataType dt, const std::vector<ROI> &rois, const ROIPoolingLayerInfo &pool_info);
    /** Compute reference softmax layer.
     *
     * @param[in] shape                Shape of the input and output tensors.
     * @param[in] dt                   Data type of input and output tensors.
     * @param[in] fixed_point_position (Optional) Number of bits for the fractional part of the fixed point numbers
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_softmax_layer(const TensorShape &shape, DataType dt, int fixed_point_position = 0);
    /** Compute reference fixed point operation.
     *
     * @param[in] shape                Shape of the input and output tensors.
     * @param[in] dt_in                Data type of the input tensor.
     * @param[in] dt_out               Data type of the output tensor.
     * @param[in] op                   Fixed point operation to perform.
     * @param[in] fixed_point_position Number of bits for the fractional part of the fixed point numbers
     *
     * @return Computed raw tensor.
     */
    static RawTensor compute_reference_fixed_point_operation(const TensorShape &shape, DataType dt_in, DataType dt_out, FixedPointOp op, int fixed_point_position);

protected:
    Reference()  = default;
    ~Reference() = default;
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif
