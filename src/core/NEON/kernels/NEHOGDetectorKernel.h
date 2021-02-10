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
#ifndef ARM_COMPUTE_NEHOGDETECTORKERNEL_H
#define ARM_COMPUTE_NEHOGDETECTORKERNEL_H

#include "arm_compute/core/IArray.h"
#include "arm_compute/core/IHOG.h"
#include "src/core/NEON/INEKernel.h"
#include "support/Mutex.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel to perform HOG detector kernel using linear SVM */
class NEHOGDetectorKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEHOGDetectorKernel";
    }
    /** Default constructor */
    NEHOGDetectorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGDetectorKernel(const NEHOGDetectorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHOGDetectorKernel &operator=(const NEHOGDetectorKernel &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEHOGDetectorKernel(NEHOGDetectorKernel &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEHOGDetectorKernel &operator=(NEHOGDetectorKernel &&) = delete;
    /** Default destructor */
    ~NEHOGDetectorKernel() = default;

    /** Initialise the kernel's input, HOG data-object, detection window, the stride of the detection window, the threshold and index of the object to detect
     *
     * @param[in]  input                   Input tensor which stores the HOG descriptor obtained with @ref NEHOGOrientationBinningKernel. Data type supported: F32. Number of channels supported: equal to the number of histogram bins per block
     * @param[in]  hog                     HOG data object used by @ref NEHOGOrientationBinningKernel and  @ref NEHOGBlockNormalizationKernel
     * @param[out] detection_windows       Array of @ref DetectionWindow. This array stores all the detected objects
     * @param[in]  detection_window_stride Distance in pixels between 2 consecutive detection windows in x and y directions.
     *                                     It must be multiple of the hog->info()->block_stride()
     * @param[in]  threshold               (Optional) Threshold for the distance between features and SVM classifying plane
     * @param[in]  idx_class               (Optional) Index of the class used for evaluating which class the detection window belongs to
     */
    void configure(const ITensor *input, const IHOG *hog, IDetectionWindowArray *detection_windows, const Size2D &detection_window_stride, float threshold = 0.0f, uint16_t idx_class = 0);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor         *_input;
    IDetectionWindowArray *_detection_windows;
    const float           *_hog_descriptor;
    float                  _bias;
    float                  _threshold;
    uint16_t               _idx_class;
    size_t                 _num_bins_per_descriptor_x;
    size_t                 _num_blocks_per_descriptor_y;
    size_t                 _block_stride_width;
    size_t                 _block_stride_height;
    size_t                 _detection_window_width;
    size_t                 _detection_window_height;
    size_t                 _max_num_detection_windows;
    arm_compute::Mutex     _mutex;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEHOGDETECTORKERNEL_H */
