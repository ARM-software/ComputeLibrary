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
#ifndef __ARM_COMPUTE_CLHOGDETECTORKERNEL_H__
#define __ARM_COMPUTE_CLHOGDETECTORKERNEL_H__

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLHOG.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/OpenCL.h"

namespace cl
{
class Buffer;
}

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform HOG detector kernel using linear SVM */
class CLHOGDetectorKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLHOGDetectorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHOGDetectorKernel(const CLHOGDetectorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHOGDetectorKernel &operator=(const CLHOGDetectorKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLHOGDetectorKernel(CLHOGDetectorKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLHOGDetectorKernel &operator=(CLHOGDetectorKernel &&) = default;
    /** Default destructor */
    ~CLHOGDetectorKernel() = default;

    /** Initialise the kernel's input, HOG data-object, detection window, the stride of the detection window, the threshold and index of the object to detect
     *
     * @param[in]  input                   Input tensor which stores the HOG descriptor obtained with @ref CLHOGOrientationBinningKernel. Data type supported: F32. Number of channels supported: equal to the number of histogram bins per block
     * @param[in]  hog                     HOG data object used by @ref CLHOGOrientationBinningKernel and  @ref CLHOGBlockNormalizationKernel
     * @param[out] detection_windows       Array of @ref DetectionWindow. This array stores all the detected objects
     * @param[in]  num_detection_windows   Number of detected objects
     * @param[in]  detection_window_stride Distance in pixels between 2 consecutive detection windows in x and y directions.
     *                                     It must be multiple of the hog->info()->block_stride()
     * @param[in]  threshold               (Optional) Threshold for the distance between features and SVM classifying plane
     * @param[in]  idx_class               (Optional) Index of the class used for evaluating which class the detection window belongs to
     */
    void configure(const ICLTensor *input, const ICLHOG *hog, ICLDetectionWindowArray *detection_windows, cl::Buffer *num_detection_windows, const Size2D &detection_window_stride, float threshold = 0.0f,
                   uint16_t idx_class = 0);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue);

private:
    const ICLTensor         *_input;
    ICLDetectionWindowArray *_detection_windows;
    cl::Buffer              *_num_detection_windows;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLHOGDETECTORKERNEL_H__ */
