/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_HOG_DETECTOR_FIXTURE
#define ARM_COMPUTE_TEST_HOG_DETECTOR_FIXTURE

#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/IHOGAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/fixtures/HOGDescriptorFixture.h"
#include "tests/validation/reference/HOGDetector.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType,
          typename HOGType,
          typename DetectionWindowArrayType,
          typename HOGDescriptorType,
          typename AccessorType,
          typename ArrayAccessorType,
          typename HOGAccessorType,
          typename HOGDetectorType,
          typename T,
          typename U>
class HOGDetectorValidationFixture : public HOGDescriptorValidationFixture<TensorType, HOGType, AccessorType, HOGDescriptorType, T, U>
{
public:
    template <typename...>
    void setup(Size2D detection_window_stride, std::string image, HOGInfo hog_info, Format format, BorderMode border_mode)
    {
        using HDF = HOGDescriptorValidationFixture<TensorType, HOGType, AccessorType, HOGDescriptorType, T, U>;
        HDF::setup(image, hog_info, format, border_mode);

        const unsigned int max_num_detection_windows = 100000;

        // Initialise descriptor (linear SVM coefficients).
        // NOTE: Fixed values are used to keep the number of detection windows detected
        // consistent in order to have meaningful validation tolerances.
        // The values are "unbalanced" to reduce the number of detected objects
        std::random_device::result_type seed       = 0;
        std::vector<U>                  descriptor = generate_random_real(hog_info.descriptor_size(), -0.505f, 0.495f, seed);

        // Compute target and reference values using feature vector from descriptor kernel
        _target    = compute_target(HDF::_target, descriptor, max_num_detection_windows, hog_info, detection_window_stride);
        _reference = compute_reference(HDF::_reference, descriptor, max_num_detection_windows, hog_info, detection_window_stride);
    }

protected:
    std::vector<DetectionWindow> compute_target(const TensorType &src, const std::vector<U> &descriptor, unsigned int max_num_detection_windows,
                                                const HOGInfo &hog_info, const Size2D &detection_window_stride)
    {
        // Create HOG
        HOGType hog = create_HOG<HOGType>(hog_info);

        // Create array of detection windows
        DetectionWindowArrayType detection_windows(max_num_detection_windows);

        // Copy HOG descriptor values to HOG memory
        {
            HOGAccessorType hog_accessor(hog);
            std::memcpy(hog_accessor.descriptor(), descriptor.data(), descriptor.size() * sizeof(U));
        }

        // Create and configure function
        HOGDetectorType hog_detector;
        hog_detector.configure(&src, &hog, &detection_windows, detection_window_stride);

        // Reset detection windows
        detection_windows.clear();

        // Compute function
        hog_detector.run();

        // Create array of detection windows
        std::vector<DetectionWindow> windows;

        // Copy detection windows
        ArrayAccessorType accessor(detection_windows);

        for(size_t i = 0; i < accessor.num_values(); i++)
        {
            DetectionWindow win;
            win.x         = accessor.at(i).x;
            win.y         = accessor.at(i).y;
            win.width     = accessor.at(i).width;
            win.height    = accessor.at(i).height;
            win.idx_class = accessor.at(i).idx_class;
            win.score     = accessor.at(i).score;

            windows.push_back(win);
        }

        return windows;
    }

    std::vector<DetectionWindow> compute_reference(const SimpleTensor<U> &src, const std::vector<U> &descriptor, unsigned int max_num_detection_windows,
                                                   const HOGInfo &hog_info, const Size2D &detection_window_stride)
    {
        // Assumes defaults value of zero for threshold and class_idx.
        return reference::hog_detector(src, descriptor, max_num_detection_windows, hog_info, detection_window_stride);
    }

    std::vector<DetectionWindow> _target{};
    std::vector<DetectionWindow> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HOG_DETECTOR_FIXTURE */
