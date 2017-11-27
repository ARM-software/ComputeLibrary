/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_HOG_DESCRIPTOR_FIXTURE
#define ARM_COMPUTE_TEST_HOG_DESCRIPTOR_FIXTURE

#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/HOGDescriptor.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename HOGType, typename AccessorType, typename FunctionType, typename T, typename U>
class HOGDescriptorValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string image, HOGInfo hog_info, Format format, BorderMode border_mode)
    {
        // Only defined borders supported
        ARM_COMPUTE_ERROR_ON(border_mode == BorderMode::UNDEFINED);

        // Generate a random constant value
        std::mt19937                     gen(library->seed());
        std::uniform_int_distribution<T> int_dist(0, 255);
        const T                          constant_border_value = int_dist(gen);

        _target    = compute_target(image, format, border_mode, constant_border_value, hog_info);
        _reference = compute_reference(image, format, border_mode, constant_border_value, hog_info);
    }

protected:
    template <typename V>
    void fill(V &&tensor, const std::string image, Format format)
    {
        library->fill(tensor, image, format);
    }

    template <typename V, typename D>
    void fill(V &&tensor, int i, D max)
    {
        library->fill_tensor_uniform(tensor, i, static_cast<D>(0), max);
    }

    TensorType compute_target(const std::string image, Format &format, BorderMode &border_mode, T constant_border_value, const HOGInfo &hog_info)
    {
        // Get image shape for src tensor
        TensorShape shape = library->get_image_shape(image);

        // Create tensor info for HOG descriptor
        TensorInfo tensor_info_hog_descriptor(hog_info, shape.x(), shape.y());

        // Create HOG
        HOGType hog = create_HOG<HOGType>(hog_info.cell_size(),
                                          hog_info.block_size(),
                                          hog_info.detection_window_size(),
                                          hog_info.block_stride(),
                                          hog_info.num_bins(),
                                          hog_info.normalization_type(),
                                          hog_info.l2_hyst_threshold(),
                                          hog_info.phase_type());

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type_from_format(format));
        TensorType dst = create_tensor<TensorType>(tensor_info_hog_descriptor.tensor_shape(), DataType::F32, tensor_info_hog_descriptor.num_channels());

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Create and configure function
        FunctionType hog_descriptor;
        hog_descriptor.configure(&src, &dst, &hog, border_mode, constant_border_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        const T max = std::numeric_limits<T>::max();

        // Fill tensors
        fill(AccessorType(src), image, format);
        fill(AccessorType(dst), 1, static_cast<U>(max));

        // Compute function
        hog_descriptor.run();

        return dst;
    }

    SimpleTensor<U> compute_reference(const std::string image, Format format, BorderMode border_mode, T constant_border_value, const HOGInfo &hog_info)
    {
        // Create reference
        SimpleTensor<T> src{ library->get_image_shape(image), data_type_from_format(format) };

        // Fill reference
        fill(src, image, format);

        return reference::hog_descriptor<U>(src, border_mode, constant_border_value, hog_info);
    }

    TensorType      _target{};
    SimpleTensor<U> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HOG_DESCRIPTOR_FIXTURE */
