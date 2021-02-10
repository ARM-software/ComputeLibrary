/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEHOGDETECTOR_H
#define ARM_COMPUTE_NEHOGDETECTOR_H

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/IHOG.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

namespace arm_compute
{
class ITensor;
class ITensorInfo;
/** Basic function to execute HOG detector based on linear SVM. This function calls the following Neon kernel:
 *
 * -# @ref NEHOGDetectorKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEHOGDetector : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEHOGDetector() = default;
    /** Prevent instances of this class from being copied */
    NEHOGDetector(const NEHOGDetector &) = delete;
    /** Default move constructor */
    NEHOGDetector(NEHOGDetector &&) = default;
    /** Prevent instances of this class from being copied */
    NEHOGDetector &operator=(const NEHOGDetector &) = delete;
    /** Default move assignment operator */
    NEHOGDetector &operator=(NEHOGDetector &&) = default;
    /** Destructor */
    ~NEHOGDetector();
    /** Initialise the kernel's input, output, HOG data object, detection window stride, threshold and index class
     *
     * @attention The function does not reset the number of values in @ref IDetectionWindowArray so it is caller's responsibility to clear it.
     *
     * @param[in]  input                   Input tensor. It is the output of @ref NEHOGDescriptor. Data type supported: F32
     * @param[in]  hog                     HOG data-object that describes the HOG descriptor
     * @param[out] detection_windows       Array of @ref DetectionWindow used to store the detected objects
     * @param[in]  detection_window_stride Distance in pixels between 2 consecutive detection windows in x and y directions.
     *                                     It must be multiple of the block stride stored in hog
     * @param[in]  threshold               (Optional) Threshold for the distance between features and SVM classifying plane
     * @param[in]  idx_class               (Optional) Index of the class used for evaluating which class the detection window belongs to
     */
    void configure(const ITensor *input, const IHOG *hog, IDetectionWindowArray *detection_windows, const Size2D &detection_window_stride, float threshold = 0.0f, size_t idx_class = 0);
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NEHOGDETECTOR_H */
