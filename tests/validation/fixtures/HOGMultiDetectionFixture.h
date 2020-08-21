/*
 * Copyright (c) 2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_HOG_MULTI_DETECTION_FIXTURE
#define ARM_COMPUTE_TEST_HOG_MULTI_DETECTION_FIXTURE

#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/IHOGAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/HOGMultiDetection.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType,
          typename HOGType,
          typename MultiHOGType,
          typename DetectionWindowArrayType,
          typename DetectionWindowStrideType,
          typename AccessorType,
          typename Size2DArrayAccessorType,
          typename DetectionWindowArrayAccessorType,
          typename HOGAccessorType,
          typename FunctionType,
          typename T,
          typename U>
class HOGMultiDetectionValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string image, std::vector<HOGInfo> models, Format format, BorderMode border_mode, bool non_maxima_suppression)
    {
        // Only defined borders supported
        ARM_COMPUTE_ERROR_ON(border_mode == BorderMode::UNDEFINED);

        // Generate a random constant value
        std::mt19937                     gen(library->seed());
        std::uniform_int_distribution<T> int_dist(0, 255);
        const T                          constant_border_value = int_dist(gen);

        // Initialize descriptors vector
        std::vector<std::vector<U>> descriptors(models.size());

        // Use default values for threshold and min_distance
        const float threshold    = 0.f;
        const float min_distance = 1.f;

        // Maximum number of detection windows per batch
        const unsigned int max_num_detection_windows = 100000;

        _target    = compute_target(image, format, border_mode, constant_border_value, models, descriptors, max_num_detection_windows, threshold, non_maxima_suppression, min_distance);
        _reference = compute_reference(image, format, border_mode, constant_border_value, models, descriptors, max_num_detection_windows, threshold, non_maxima_suppression, min_distance);
    }

protected:
    template <typename V>
    void fill(V &&tensor, const std::string image, Format format)
    {
        library->fill(tensor, image, format);
    }

    void initialize_batch(const std::vector<HOGInfo> &models, MultiHOGType &multi_hog,
                          std::vector<std::vector<U>> &descriptors, DetectionWindowStrideType &detection_window_strides)
    {
        for(unsigned i = 0; i < models.size(); ++i)
        {
            auto hog_model = reinterpret_cast<HOGType *>(multi_hog.model(i));
            hog_model->init(models[i]);

            // Initialise descriptor (linear SVM coefficients).
            std::random_device::result_type seed = 0;
            descriptors.at(i)                    = generate_random_real(models[i].descriptor_size(), -0.505f, 0.495f, seed);

            // Copy HOG descriptor values to HOG memory
            {
                HOGAccessorType hog_accessor(*hog_model);
                std::memcpy(hog_accessor.descriptor(), descriptors.at(i).data(), descriptors.at(i).size() * sizeof(U));
            }

            // Initialize detection window stride
            Size2DArrayAccessorType accessor(detection_window_strides);
            accessor.at(i) = models[i].block_stride();
        }
    }

    std::vector<DetectionWindow> compute_target(const std::string image, Format &format, BorderMode &border_mode, T constant_border_value,
                                                const std::vector<HOGInfo> &models, std::vector<std::vector<U>> &descriptors, unsigned int max_num_detection_windows,
                                                float threshold, bool non_max_suppression, float min_distance)
    {
        MultiHOGType              multi_hog(models.size());
        DetectionWindowArrayType  detection_windows(max_num_detection_windows);
        DetectionWindowStrideType detection_window_strides(models.size());

        // Resize detection window_strides for index access
        detection_window_strides.resize(models.size());

        // Initialiize MultiHOG and detection windows
        initialize_batch(models, multi_hog, descriptors, detection_window_strides);

        // Get image shape for src tensor
        TensorShape shape = library->get_image_shape(image);

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type_from_format(format));
        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Create and configure function
        FunctionType hog_multi_detection;
        hog_multi_detection.configure(&src, &multi_hog, &detection_windows, &detection_window_strides, border_mode, constant_border_value, threshold, non_max_suppression, min_distance);

        // Reset detection windows
        detection_windows.clear();

        // Allocate tensors
        src.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), image, format);

        // Compute function
        hog_multi_detection.run();

        // Copy detection windows
        std::vector<DetectionWindow>     windows;
        DetectionWindowArrayAccessorType accessor(detection_windows);

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

    std::vector<DetectionWindow> compute_reference(const std::string image, Format format, BorderMode border_mode, T constant_border_value,
                                                   const std::vector<HOGInfo> &models, const std::vector<std::vector<U>> &descriptors, unsigned int max_num_detection_windows,
                                                   float threshold, bool non_max_suppression, float min_distance)
    {
        // Create reference
        SimpleTensor<T> src{ library->get_image_shape(image), data_type_from_format(format) };

        // Fill reference
        fill(src, image, format);

        // NOTE: Detection window stride fixed to block stride
        return reference::hog_multi_detection(src, border_mode, constant_border_value, models, descriptors, max_num_detection_windows, threshold, non_max_suppression, min_distance);
    }

    std::vector<DetectionWindow> _target{};
    std::vector<DetectionWindow> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HOG_MULTI_DETECTION_FIXTURE */
