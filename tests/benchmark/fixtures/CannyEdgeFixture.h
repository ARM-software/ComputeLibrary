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
#ifndef ARM_COMPUTE_TEST_CANNY_EDGE_FIXTURE
#define ARM_COMPUTE_TEST_CANNY_EDGE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
class CLCannyEdge;
class NECannyEdge;

namespace test
{
namespace benchmark
{
template <typename TensorType, typename Function, typename Accessor>
class CannyEdgeFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string image, int gradient_size, MagnitudeType norm_type, BorderMode border_mode, bool use_fp16, Format format)
    {
        // Load the image (cached by the library if loaded before)
        const RawTensor &raw = library->get(image, format);

        src = create_tensor<TensorType>(raw.shape(), format);
        dst = create_tensor<TensorType>(raw.shape(), format);

        configure_target<Function>(canny_edge_func, src, dst, gradient_size, static_cast<int>(norm_type) + 1, border_mode, use_fp16);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        library->fill(Accessor(src), raw);
    }

    void run()
    {
        canny_edge_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

protected:
    template <typename F, typename std::enable_if<std::is_same<F, NECannyEdge>::value, int>::type = 0>
    void configure_target(F &func, TensorType &src, TensorType &dst, int gradient_size, int norm_type, BorderMode border_mode, bool use_fp16)
    {
        func.configure(&src, &dst, upper_thresh, lower_thresh, gradient_size, norm_type, border_mode, constant_border_value, use_fp16);
    }

    template <typename F, typename std::enable_if<std::is_same<F, CLCannyEdge>::value, int>::type = 0>
    void configure_target(F &func, TensorType &src, TensorType &dst, int gradient_size, int norm_type, BorderMode border_mode, bool use_fp16)
    {
        ARM_COMPUTE_UNUSED(use_fp16);
        ARM_COMPUTE_ERROR_ON(use_fp16);

        func.configure(&src, &dst, upper_thresh, lower_thresh, gradient_size, norm_type, border_mode, constant_border_value);
    }

private:
    static const int32_t lower_thresh          = 0;
    static const int32_t upper_thresh          = 255;
    static const uint8_t constant_border_value = 0;

    TensorType src{};
    TensorType dst{};
    Function   canny_edge_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CANNY_EDGE_FIXTURE */
