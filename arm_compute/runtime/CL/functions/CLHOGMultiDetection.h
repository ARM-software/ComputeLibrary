/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLHOGMULTIDETECTION_H__
#define __ARM_COMPUTE_CLHOGMULTIDETECTION_H__

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLMultiHOG.h"
#include "arm_compute/core/CL/kernels/CLHOGDescriptorKernel.h"
#include "arm_compute/core/CPP/kernels/CPPDetectionWindowNonMaximaSuppressionKernel.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLHOGDetector.h"
#include "arm_compute/runtime/CL/functions/CLHOGGradient.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
/** Basic function to detect multiple objects (or the same object at different scales) on the same input image using HOG. This function calls the following kernels:
 *
 * -# @ref CLHOGGradient
 * -# @ref CLHOGOrientationBinningKernel
 * -# @ref CLHOGBlockNormalizationKernel
 * -# @ref CLHOGDetector
 * -# @ref CPPDetectionWindowNonMaximaSuppressionKernel (executed if non_maxima_suppression == true)
 *
 * @note This implementation works if all the HOG data-objects within the IMultiHOG container have the same:
 *       -# Phase type
         -# Normalization type
         -# L2 hysteresis threshold if the normalization type is L2HYS_NORM
 *
 */
class CLHOGMultiDetection : public IFunction
{
public:
    /** Default constructor */
    CLHOGMultiDetection(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHOGMultiDetection(const CLHOGMultiDetection &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHOGMultiDetection &operator=(const CLHOGMultiDetection &) = delete;
    /** Initialise the function's source, destination, detection window strides, border mode, threshold and non-maxima suppression
     *
     * @param[in, out] input                    Input tensor. Data type supported: U8
     *                                          (Written to only for @p border_mode != UNDEFINED)
     * @param[in]      multi_hog                Container of multiple HOG data object. Each HOG data object describes one HOG model to detect.
     *                                          This container should store the HOG data-objects in descending or ascending cell_size width order.
     *                                          This will help to understand if the HOG descriptor computation can be skipped for some HOG data-objects
     * @param[out]     detection_windows        Array of @ref DetectionWindow used for locating the detected objects
     * @param[in]      detection_window_strides Array of @ref Size2D used to specify the distance in pixels between 2 consecutive detection windows in x and y directions for each HOG data-object
     *                                          The dimension of this array must be the same of multi_hog->num_models()
     *                                          The i-th detection_window_stride of this array must be multiple of the block_stride stored in the i-th multi_hog array
     * @param[in]      border_mode              Border mode to use.
     * @param[in]      constant_border_value    (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     * @param[in]      threshold                (Optional) Threshold for the distance between features and SVM classifying plane
     * @param[in]      non_maxima_suppression   (Optional) Flag to specify whether the non-maxima suppression is required or not.
     *                                          True if the non-maxima suppression stage has to be computed
     * @param[in]      min_distance             (Optional) Radial Euclidean distance to use for the non-maxima suppression stage
     *
     */
    void configure(ICLTensor *input, const ICLMultiHOG *multi_hog, ICLDetectionWindowArray *detection_windows, ICLSize2DArray *detection_window_strides, BorderMode border_mode,
                   uint8_t constant_border_value = 0,
                   float threshold = 0.0f, bool non_maxima_suppression = false, float min_distance = 1.0f);

    // Inherited method overridden:
    void run() override;

private:
    CLMemoryGroup                                _memory_group;
    CLHOGGradient                                _gradient_kernel;
    std::vector<CLHOGOrientationBinningKernel>   _orient_bin_kernel;
    std::vector<CLHOGBlockNormalizationKernel>   _block_norm_kernel;
    std::vector<CLHOGDetector>                   _hog_detect_kernel;
    CPPDetectionWindowNonMaximaSuppressionKernel _non_maxima_kernel;
    std::vector<CLTensor>                        _hog_space;
    std::vector<CLTensor>                        _hog_norm_space;
    ICLDetectionWindowArray                     *_detection_windows;
    CLTensor                                     _mag;
    CLTensor                                     _phase;
    bool                                         _non_maxima_suppression;
    size_t                                       _num_orient_bin_kernel;
    size_t                                       _num_block_norm_kernel;
    size_t                                       _num_hog_detect_kernel;
};
}

#endif /* __ARM_COMPUTE_CLHOGMULTIDETECTION_H__ */