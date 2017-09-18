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

#include "RawTensor.h"
#include "Reference.h"

#include <map>
#include <memory>
#include <ostream>
#include <vector>

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
    /** Function to compute reference Harris corners.
     *
     * @param[in]  src                   Input tensor
     * @param[in]  Gx                    Tensor used to compute Sobel along the x axis
     * @param[in]  Gy                    Tensor used to compute Sobel along the y axis
     * @param[in]  candidates            Tensor used to store candidate corners
     * @param[in]  non_maxima            Tensor used to store non_maxima suppressed candidate corners
     * @param[in]  threshold             Minimum threshold with which to eliminate Harris Corner scores (computed using the normalized Sobel kernel).
     * @param[in]  min_dist              Radial Euclidean distance for the euclidean distance stage
     * @param[in]  sensitivity           Sensitivity threshold k from the Harris-Stephens equation
     * @param[in]  gradient_size         The gradient window size to use on the input. The implementation supports 3, 5, and 7
     * @param[in]  block_size            The block window size used to compute the Harris Corner score. The implementation supports 3, 5, and 7.
     * @param[out] corners               Array of keypoints to store the results.
     * @param[in]  border_mode           Border mode to use
     * @param[in]  constant_border_value Constant value to use for borders if border_mode is set to CONSTANT.
     *
     */
    static void harris_corners(RawTensor &src, RawTensor &Gx, RawTensor &Gy, const RawTensor &candidates, const RawTensor &non_maxima, float threshold, float min_dist, float sensitivity,
                               int32_t gradient_size, int32_t block_size, KeyPointArray &corners, BorderMode border_mode, uint8_t constant_border_value);
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
    /** Compute non linear filter function.
     *
     * @param[in]  src                   First input tensor
     * @param[out] dst                   Output tensor
     * @param[in]  function              Non linear function to perform
     * @param[in]  mask_size             Mask size. Supported sizes: 3, 5
     * @param[in]  pattern               Matrix pattern
     * @param[in]  mask                  The given mask.
     * @param[in]  border_mode           Strategy to use for borders.
     * @param[in]  constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
    */
    static void non_linear_filter(const RawTensor &src, RawTensor &dst, NonLinearFilterFunction function, unsigned int mask_size,
                                  MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode, uint8_t constant_border_value = 0);
    /** Threshold of@p src to @p dst
     *
     * @param[in]  src         Input tensor.
     * @param[out] dst         Result tensor.
     * @param[in]  threshold   Threshold. When the threhold type is RANGE, this is used as the lower threshold.
     * @param[in]  false_value value to set when the condition is not respected.
     * @param[in]  true_value  value to set when the condition is respected.
     * @param[in]  type        Thresholding type. Either RANGE or BINARY.
     * @param[in]  upper       Upper threshold. Only used when the thresholding type is RANGE.
     */
    static void threshold(const RawTensor &src, RawTensor &dst, uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper);
    /** ROI Pooling layer of @p src based on the information from @p pool_info and @p rois.
     *
     * @param[in]  src       Input tensor.
     * @param[out] dst       Result tensor.
     * @param[in]  rois      Region of Interest points.
     * @param[in]  pool_info ROI Pooling Layer information.
     */
    static void roi_pooling_layer(const RawTensor &src, RawTensor &dst, const std::vector<ROI> &rois, const ROIPoolingLayerInfo &pool_info);

private:
    ReferenceCPP()  = delete;
    ~ReferenceCPP() = delete;
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_REFERENCE_REFERENCE_CPP_H__ */
