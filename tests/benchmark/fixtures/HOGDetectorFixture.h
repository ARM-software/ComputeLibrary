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

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/benchmark/fixtures/HOGDescriptorFixture.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType,
          typename HOGType,
          typename Function,
          typename Accessor,
          typename HOGAccessor,
          typename HOGDescriptorType,
          typename ArrayType>
class HOGDetectorFixture : public HOGDescriptorFixture<TensorType, HOGType, HOGDescriptorType, Accessor>
{
public:
    template <typename...>
    void setup(Size2D detection_window_stride, std::string image, HOGInfo hog_info, Format format, BorderMode border_mode)
    {
        HDF::setup(image, hog_info, format, border_mode);
        HDF::run();

        // Initialise descriptor (linear SVM coefficients).
        // NOTE: Fixed values are used to keep the number of detection windows detected
        // consistent in order to have meaningful validation tolerances.
        // The values are "unbalanced" to reduce the number of detected objects
        const std::random_device::result_type seed       = 0;
        std::vector<float>                    descriptor = generate_random_real(hog_info.descriptor_size(), -0.505f, 0.495f, seed);

        // Create HOG
        hog = create_HOG<HOGType>(hog_info);

        // Copy HOG descriptor values to HOG memory
        {
            HOGAccessor hog_accessor(hog);
            std::memcpy(hog_accessor.descriptor(), descriptor.data(), descriptor.size() * sizeof(float));
        }

        // Create and configure function
        hog_detector_func.configure(&(HDF::dst), &hog, &detection_windows, detection_window_stride);

        // Reset detection windows
        detection_windows.clear();
    }

    void run()
    {
        hog_detector_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
    }

private:
    static const unsigned int max_num_detection_windows = 100000;

    HOGType   hog{};
    Function  hog_detector_func{};
    ArrayType detection_windows{ max_num_detection_windows };

    using HDF = HOGDescriptorFixture<TensorType, HOGType, HOGDescriptorType, Accessor>;
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HOG_DETECTOR_FIXTURE */
