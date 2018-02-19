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
#ifndef ARM_COMPUTE_TEST_HOG_DESCRIPTOR_FIXTURE
#define ARM_COMPUTE_TEST_HOG_DESCRIPTOR_FIXTURE

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
template <typename TensorType, typename HOGType, typename Function, typename Accessor>
class HOGDescriptorFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string image, HOGInfo hog_info, Format format, BorderMode border_mode)
    {
        std::mt19937                           generator(library->seed());
        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        uint8_t                                constant_border_value = static_cast<uint8_t>(distribution_u8(generator));

        // Load the image (cached by the library if loaded before)
        const RawTensor &raw = library->get(image, format);

        // Create tensor info for HOG descriptor
        TensorInfo tensor_info_hog_descriptor(hog_info, raw.shape().x(), raw.shape().y());

        // Create tensor
        src = create_tensor<TensorType>(raw.shape(), format);
        dst = create_tensor<TensorType>(tensor_info_hog_descriptor.tensor_shape(),
                                        DataType::F32, tensor_info_hog_descriptor.num_channels());

        // Create HOG
        HOGType hog = create_HOG<HOGType>(hog_info);

        // Create and configure function
        hog_descriptor_func.configure(&src, &dst, &hog, border_mode, constant_border_value);

        // Allocate tensor
        src.allocator()->allocate();
        dst.allocator()->allocate();

        // Copy image data to tensor
        library->fill(Accessor(src), raw);
    }

    void run()
    {
        hog_descriptor_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
    }

    void teardown()
    {
        src.allocator()->free();
        dst.allocator()->free();
    }

private:
    TensorType src{};
    TensorType dst{};
    Function   hog_descriptor_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HOG_DESCRIPTOR_FIXTURE */
