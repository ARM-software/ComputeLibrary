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
#ifndef ARM_COMPUTE_TEST_HARRIS_CORNERS_FIXTURE
#define ARM_COMPUTE_TEST_HARRIS_CORNERS_FIXTURE

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
template <typename TensorType, typename Function, typename Accessor, typename ArrayType>
class HarrisCornersFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string image, Format format, float threshold, float min_dist, float sensitivity,
               int32_t gradient_size, int32_t block_size,
               BorderMode border_mode, bool use_fp16)
    {
        // Load the image (cached by the library if loaded before)
        const RawTensor &raw = library->get(image, format);

        // Create tensor
        src = create_tensor<TensorType>(raw.shape(), format);

        // Create and configure function
        harris_corners_func.configure(&src, threshold, min_dist, sensitivity, gradient_size, block_size, &out, border_mode, 0, use_fp16);

        // Allocate tensor
        src.allocator()->allocate();

        // Copy image data to tensor
        library->fill(Accessor(src), raw);
    }

    void run()
    {
        harris_corners_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
    }

    void teardown()
    {
        src.allocator()->free();
    }

private:
    TensorType src{};
    ArrayType  out{ 20000 };
    Function   harris_corners_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HARRIS_CORNERS_FIXTURE */