/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_HOG_DETECTOR_H
#define ARM_COMPUTE_TEST_HOG_DETECTOR_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "tests/SimpleTensor.h"

#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
std::vector<DetectionWindow> hog_detector(const SimpleTensor<T> &src, const std::vector<T> &descriptor, unsigned int max_num_detection_windows,
                                          const HOGInfo &hog_info, const Size2D &detection_window_stride, float threshold = 0.0f, uint16_t idx_class = 0);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HOG_DETECTOR_H */
