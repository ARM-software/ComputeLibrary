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
#ifndef ARM_COMPUTE_TEST_HOG_MULTI_DETECTION_FIXTURE
#define ARM_COMPUTE_TEST_HOG_MULTI_DETECTION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType,
          typename HOGType,
          typename MultiHOGType,
          typename DetectionWindowArrayType,
          typename DetectionWindowStrideType,
          typename Function,
          typename Accessor,
          typename HOGAccessorType,
          typename Size2DArrayAccessorType>
class HOGMultiDetectionFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string image, std::vector<HOGInfo> models, Format format, BorderMode border_mode, bool non_maxima_suppression)
    {
        // Only defined borders supported
        ARM_COMPUTE_ERROR_ON(border_mode == BorderMode::UNDEFINED);

        std::mt19937                           generator(library->seed());
        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        uint8_t                                constant_border_value = static_cast<uint8_t>(distribution_u8(generator));

        // Load the image (cached by the library if loaded before)
        const RawTensor &raw = library->get(image, format);

        // Initialize descriptors vector
        std::vector<std::vector<float>> descriptors(models.size());

        // Resize detection window_strides for index access
        detection_window_strides.resize(models.size());

        // Initialiize MultiHOG and detection windows
        initialize_batch(models, multi_hog, descriptors, detection_window_strides);

        // Create tensors
        src = create_tensor<TensorType>(raw.shape(), format);

        // Use default values for threshold and min_distance
        const float threshold    = 0.f;
        const float min_distance = 1.f;

        hog_multi_detection_func.configure(&src,
                                           &multi_hog,
                                           &detection_windows,
                                           &detection_window_strides,
                                           border_mode,
                                           constant_border_value,
                                           threshold,
                                           non_maxima_suppression,
                                           min_distance);

        // Reset detection windows
        detection_windows.clear();

        // Allocate tensor
        src.allocator()->allocate();

        library->fill(Accessor(src), raw);
    }

    void run()
    {
        hog_multi_detection_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
    }

private:
    void initialize_batch(const std::vector<HOGInfo> &models, MultiHOGType &multi_hog,
                          std::vector<std::vector<float>> &descriptors, DetectionWindowStrideType &detection_window_strides)
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
                std::memcpy(hog_accessor.descriptor(), descriptors.at(i).data(), descriptors.at(i).size() * sizeof(float));
            }

            // Initialize detection window stride
            Size2DArrayAccessorType accessor(detection_window_strides);
            accessor.at(i) = models[i].block_stride();
        }
    }

private:
    static const unsigned int model_size                = 4;
    static const unsigned int max_num_detection_windows = 100000;

    MultiHOGType              multi_hog{ model_size };
    DetectionWindowStrideType detection_window_strides{ model_size };
    DetectionWindowArrayType  detection_windows{ max_num_detection_windows };

    TensorType src{};
    Function   hog_multi_detection_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HOG_MULTI_DETECTION_FIXTURE */
