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
#include "arm_compute/runtime/CL/functions/CLHOGDetector.h"

#include "arm_compute/core/CL/kernels/CLHOGDetectorKernel.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <algorithm>

using namespace arm_compute;

CLHOGDetector::CLHOGDetector()
    : _hog_detector_kernel(), _detection_windows(nullptr), _num_detection_windows()
{
}

void CLHOGDetector::configure(const ICLTensor *input, const ICLHOG *hog, ICLDetectionWindowArray *detection_windows, const Size2D &detection_window_stride, float threshold, size_t idx_class)
{
    _detection_windows = detection_windows;

    // Allocate buffer for storing the number of detected objects
    _num_detection_windows = cl::Buffer(CLScheduler::get().context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(unsigned int));

    // Configure HOGDetectorKernel
    _hog_detector_kernel.configure(input, hog, detection_windows, &_num_detection_windows, detection_window_stride, threshold, idx_class);
}

void CLHOGDetector::run()
{
    cl::CommandQueue q = CLScheduler::get().queue();

    // Reset number of detections
    const unsigned int init_num_detection_windows = _detection_windows->num_values();
    q.enqueueWriteBuffer(_num_detection_windows, CL_FALSE, 0, sizeof(unsigned int), &init_num_detection_windows);

    // Run CLHOGDetectorKernel
    CLScheduler::get().enqueue(_hog_detector_kernel);

    // Read number of detections
    unsigned int num_detection_windows = 0;
    q.enqueueReadBuffer(_num_detection_windows, CL_TRUE, 0, sizeof(unsigned int), &num_detection_windows);

    // Update the number of values stored in _detection_windows
    _detection_windows->resize(static_cast<size_t>(num_detection_windows));

    q.flush();
}