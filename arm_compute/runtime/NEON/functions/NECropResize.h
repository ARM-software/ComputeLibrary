/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEON_CROP_RESIZE_H
#define ARM_COMPUTE_NEON_CROP_RESIZE_H

#include "arm_compute/runtime/NEON/functions/NEScale.h"

#include <memory>

namespace arm_compute
{
// Forward Declarations
class Tensor;
class ITensor;
class NECropKernel;

/** Function to perform cropping and resizing */
class NECropResize : public IFunction
{
public:
    /** Default constructor */
    NECropResize();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECropResize(const NECropResize &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECropResize &operator=(const NECropResize &) = delete;
    /** Allow instances of this class to be moved */
    NECropResize(NECropResize &&) = default;
    /** Allow instances of this class to be moved */
    NECropResize &operator=(NECropResize &&) = default;
    /** Default destructor */
    ~NECropResize();

    /** Configure kernel
     *
     * Valid data layouts:
     * - NHWC
     *
     * Valid data type configurations:
     * |src0     |src1     |src2   |dst      |
     * |:--------|:--------|:------|:--------|
     * |All      |F32      |F32    |F32      |
     *
     * @note Supported tensor rank: up to 4
     * @note Box indices may be outside of the bounds, in which case @p extrapolation_value is used.
     * @note Start and end indices of boxes are inclusive.
     *
     * @param[in]  input               Source tensor containing N batches of 3D images to be cropped. Data type supported: U8/U16/S16/U32/S32/F16/F32
     * @param[in]  boxes               Tensor containing the boxes used to crop the images. Data type supported: F32
     * @param[in]  box_ind             One dimensional tensor containing the batch index of the 3D image in @p input that the corresponding
     *                                 box in @p boxes will be applied to. Data type supported: F32
     * @param[out] output              Destination tensor containing a cropped and resized image for each box in @p boxes. Data type supported: F32
     * @param[in]  crop_size           The dimensions that each cropped image will be resized to.
     * @param[in]  method              The policy to be used when resizing image. Default is bilinear.
     * @param[in]  extrapolation_value Value to be used for values outside of the image for cropping and resizing. Default is 0.
     */
    void configure(const ITensor *input, const ITensor *boxes, const ITensor *box_ind, ITensor *output, Coordinates2D crop_size,
                   InterpolationPolicy method = InterpolationPolicy::BILINEAR, float extrapolation_value = 0);

    /** Static function to check if given info will lead to a valid configuration of @ref NESlice
     *
     * @note Supported tensor rank: up to 4
     * @note Box indices may be outside of the bounds, in which case @p extrapolation_value is used.
     * @note Start and end indices of boxes are inclusive.
     *
     * @param[in] input               Source tensor containing N batches of 3D images to be cropped. Data type supported: U8/U16/S16/U32/S32/F16/F32
     * @param[in] boxes               Tensor info for the tensor containing the boxes used to crop the images. Data type supported: F32
     * @param[in] box_ind             Tensor info for the one dimensional tensor containing the batch index of the 3D image in @p input
     *                                that the corresponding box in @p boxes will be applied to. Data type supported: F32
     * @param[in] output              Tensor info for the destination tensor containing a cropped and resized image for each box in @p boxes.
     *                                Data type supported: F32
     * @param[in] crop_size           The dimensions that each cropped image will be resized to.
     * @param[in] method              The policy to be used when resizing image. Default is bilinear.
     * @param[in] extrapolation_value Value to be used for values outside of the image for cropping and resizing. Default is 0.
     *
     * @return A status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *boxes, const ITensorInfo *box_ind, const ITensorInfo *output,
                           Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value);

    void run() override;

    ITensor            *_output;
    size_t              _num_boxes;
    InterpolationPolicy _method;
    float               _extrapolation_value;

    std::vector<std::unique_ptr<NECropKernel>> _crop;
    std::vector<std::unique_ptr<NEScale>>      _scale;
    std::vector<std::unique_ptr<Tensor>>       _crop_results;
    std::vector<std::unique_ptr<Tensor>>       _scaled_results;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEON_CROP_RESIZE_H */
