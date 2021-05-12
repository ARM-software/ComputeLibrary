/*
* Copyright (c) 2020-2021 Arm Limited.
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
#ifndef SRC_CORE_HELPERS_SCALEHELPERS_H
#define SRC_CORE_HELPERS_SCALEHELPERS_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/QuantizationInfo.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace scale_helpers
{
/** Computes bilinear interpolation for quantized input and output, using the pointer to the top-left pixel and the pixel's distance between
 * the real coordinates and the smallest following integer coordinates. Input must be QASYMM8 and in single channel format.
 *
 * @param[in] pixel_ptr Pointer to the top-left pixel value of a single channel input.
 * @param[in] stride    Stride to access the bottom-left and bottom-right pixel values
 * @param[in] dx        Pixel's distance between the X real coordinate and the smallest X following integer
 * @param[in] dy        Pixel's distance between the Y real coordinate and the smallest Y following integer
 * @param[in] iq_info   Input QuantizationInfo
 * @param[in] oq_info   Output QuantizationInfo
 *
 * @note dx and dy must be in the range [0, 1.0]
 *
 * @return The bilinear interpolated pixel value
 */
inline uint8_t delta_bilinear_c1_quantized(const uint8_t *pixel_ptr, size_t stride, float dx, float dy,
                                           UniformQuantizationInfo iq_info, UniformQuantizationInfo oq_info)
{
    ARM_COMPUTE_ERROR_ON(pixel_ptr == nullptr);

    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;

    const float a00 = dequantize_qasymm8(*pixel_ptr, iq_info);
    const float a01 = dequantize_qasymm8(*(pixel_ptr + 1), iq_info);
    const float a10 = dequantize_qasymm8(*(pixel_ptr + stride), iq_info);
    const float a11 = dequantize_qasymm8(*(pixel_ptr + stride + 1), iq_info);

    const float w1  = dx1 * dy1;
    const float w2  = dx * dy1;
    const float w3  = dx1 * dy;
    const float w4  = dx * dy;
    float       res = a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4;
    return static_cast<uint8_t>(quantize_qasymm8(res, oq_info));
}

/** Computes bilinear interpolation for quantized input and output, using the pointer to the top-left pixel and the pixel's distance between
 * the real coordinates and the smallest following integer coordinates. Input must be QASYMM8_SIGNED and in single channel format.
 *
 * @param[in] pixel_ptr Pointer to the top-left pixel value of a single channel input.
 * @param[in] stride    Stride to access the bottom-left and bottom-right pixel values
 * @param[in] dx        Pixel's distance between the X real coordinate and the smallest X following integer
 * @param[in] dy        Pixel's distance between the Y real coordinate and the smallest Y following integer
 * @param[in] iq_info   Input QuantizationInfo
 * @param[in] oq_info   Output QuantizationInfo
 *
 * @note dx and dy must be in the range [0, 1.0]
 *
 * @return The bilinear interpolated pixel value
 */
inline int8_t delta_bilinear_c1_quantized(const int8_t *pixel_ptr, size_t stride, float dx, float dy,
                                          UniformQuantizationInfo iq_info, UniformQuantizationInfo oq_info)
{
    ARM_COMPUTE_ERROR_ON(pixel_ptr == nullptr);

    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;

    const float a00 = dequantize_qasymm8_signed(*pixel_ptr, iq_info);
    const float a01 = dequantize_qasymm8_signed(*(pixel_ptr + 1), iq_info);
    const float a10 = dequantize_qasymm8_signed(*(pixel_ptr + stride), iq_info);
    const float a11 = dequantize_qasymm8_signed(*(pixel_ptr + stride + 1), iq_info);

    const float w1  = dx1 * dy1;
    const float w2  = dx * dy1;
    const float w3  = dx1 * dy;
    const float w4  = dx * dy;
    float       res = a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4;
    return static_cast<int8_t>(quantize_qasymm8_signed(res, oq_info));
}

/** Return the pixel at (x,y) using area interpolation by clamping when out of borders. The image must be single channel U8
 *
 * @note The interpolation area depends on the width and height ration of the input and output images
 * @note Currently average of the contributing pixels is calculated
 *
 * @param[in] first_pixel_ptr Pointer to the first pixel of a single channel U8 image.
 * @param[in] stride          Stride in bytes of the image
 * @param[in] width           Width of the image
 * @param[in] height          Height of the image
 * @param[in] wr              Width ratio among the input image width and output image width.
 * @param[in] hr              Height ratio among the input image height and output image height.
 * @param[in] x               X position of the wanted pixel
 * @param[in] y               Y position of the wanted pixel
 *
 * @return The pixel at (x, y) using area interpolation.
 */
inline uint8_t
pixel_area_c1u8_clamp(const uint8_t *first_pixel_ptr, size_t stride, size_t width, size_t height, float wr,
                      float hr, int x, int y)
{
    ARM_COMPUTE_ERROR_ON(first_pixel_ptr == nullptr);

    // Calculate sampling position
    float in_x = (x + 0.5f) * wr - 0.5f;
    float in_y = (y + 0.5f) * hr - 0.5f;

    // Get bounding box offsets
    int x_from = std::floor(x * wr - 0.5f - in_x);
    int y_from = std::floor(y * hr - 0.5f - in_y);
    int x_to   = std::ceil((x + 1) * wr - 0.5f - in_x);
    int y_to   = std::ceil((y + 1) * hr - 0.5f - in_y);

    // Clamp position to borders
    in_x = std::max(-1.f, std::min(in_x, static_cast<float>(width)));
    in_y = std::max(-1.f, std::min(in_y, static_cast<float>(height)));

    // Clamp bounding box offsets to borders
    x_from = ((in_x + x_from) < -1) ? -1 : x_from;
    y_from = ((in_y + y_from) < -1) ? -1 : y_from;
    x_to   = ((in_x + x_to) > width) ? (width - in_x) : x_to;
    y_to   = ((in_y + y_to) > height) ? (height - in_y) : y_to;

    // Get pixel index
    const int xi = std::floor(in_x);
    const int yi = std::floor(in_y);

    // Bounding box elements in each dimension
    const int x_elements = (x_to - x_from + 1);
    const int y_elements = (y_to - y_from + 1);
    ARM_COMPUTE_ERROR_ON(x_elements == 0 || y_elements == 0);

    // Sum pixels in area
    int sum = 0;
    for(int j = yi + y_from, je = yi + y_to; j <= je; ++j)
    {
        const uint8_t *ptr = first_pixel_ptr + j * stride + xi + x_from;
        sum                = std::accumulate(ptr, ptr + x_elements, sum);
    }

    // Return average
    return sum / (x_elements * y_elements);
}

/** Computes bilinear interpolation using the top-left, top-right, bottom-left, bottom-right pixels and the pixel's distance between
 * the real coordinates and the smallest following integer coordinates.
 *
 * @param[in] a00    The top-left pixel value.
 * @param[in] a01    The top-right pixel value.
 * @param[in] a10    The bottom-left pixel value.
 * @param[in] a11    The bottom-right pixel value.
 * @param[in] dx_val Pixel's distance between the X real coordinate and the smallest X following integer
 * @param[in] dy_val Pixel's distance between the Y real coordinate and the smallest Y following integer
 *
 * @note dx and dy must be in the range [0, 1.0]
 *
 * @return The bilinear interpolated pixel value
 */
inline float delta_bilinear(float a00, float a01, float a10, float a11, float dx_val, float dy_val)
{
    const float dx1_val = 1.0f - dx_val;
    const float dy1_val = 1.0f - dy_val;

    const float w1 = dx1_val * dy1_val;
    const float w2 = dx_val * dy1_val;
    const float w3 = dx1_val * dy_val;
    const float w4 = dx_val * dy_val;
    return a00 * w1 + a01 * w2 + a10 * w3 + a11 * w4;
}
} // namespace scale_helpers
} // namespace arm_compute

#endif /* SRC_CORE_HELPERS_SCALEHELPERS_H */
