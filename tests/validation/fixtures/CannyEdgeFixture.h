/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/CannyEdgeDetector.h"

namespace arm_compute
{
class CLCannyEdge;
class NECannyEdge;

namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename ArrayType, typename FunctionType, typename T>
class CannyEdgeValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(std::string image, int gradient_size, MagnitudeType norm_type, BorderMode border_mode, bool use_fp16, Format format)
    {
        CannyEdgeParameters params = canny_edge_parameters();

        _target = compute_target(image, gradient_size, norm_type, border_mode, use_fp16, format, params);
        _reference = compute_reference(image, gradient_size, norm_type, border_mode, format, params);
    }

protected:
    template <typename U>
    void fill(U &&tensor, RawTensor raw)
    {
        library->fill(tensor, raw);
    }

    template <typename F, typename std::enable_if<std::is_same<F, NECannyEdge>::value, int>::type = 0>
    void configure_target(F &func, TensorType &src, TensorType &dst, int gradient_size, int norm_type, BorderMode border_mode, bool use_fp16, const CannyEdgeParameters &params)
    {
        func.configure(&src, &dst, params.upper_thresh, params.lower_thresh, gradient_size, norm_type, border_mode, params.constant_border_value, use_fp16);
    }

    template <typename F, typename std::enable_if<std::is_same<F, CLCannyEdge>::value, int>::type = 0>
    void configure_target(F &func, TensorType &src, TensorType &dst, int gradient_size, int norm_type, BorderMode border_mode, bool use_fp16, const CannyEdgeParameters &params)
    {
        ARM_COMPUTE_UNUSED(use_fp16);
        ARM_COMPUTE_ERROR_ON(use_fp16);
        func.configure(&src, &dst, params.upper_thresh, params.lower_thresh, gradient_size, norm_type, border_mode, params.constant_border_value);
    }

    TensorType compute_target(const std::string &image, int gradient_size, MagnitudeType norm_type, BorderMode border_mode, bool use_fp16, Format format, const CannyEdgeParameters &params)
    {
        // Load the image (cached by the library if loaded before)
        const RawTensor &raw = library->get(image, format);

        // Create tensors
        TensorType src = create_tensor<TensorType>(raw.shape(), format);
        TensorType dst = create_tensor<TensorType>(raw.shape(), format);
        src.info()->set_format(format);
        dst.info()->set_format(format);

        // Create Canny edge configure function
        FunctionType canny_edge;
        configure_target<FunctionType>(canny_edge, src, dst, gradient_size, static_cast<int>(norm_type) + 1, border_mode, use_fp16, params);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), raw);

        // Compute function
        canny_edge.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const std::string &image, int gradient_size, MagnitudeType norm_type, BorderMode border_mode, Format format, const CannyEdgeParameters &params)
    {
        ARM_COMPUTE_ERROR_ON(format != Format::U8);

        // Load the image (cached by the library if loaded before)
        const RawTensor &raw = library->get(image, format);

        // Create reference
        SimpleTensor<T> src{ raw.shape(), format };

        // Fill reference
        fill(src, raw);

        return reference::canny_edge_detector<T>(src, params.upper_thresh, params.lower_thresh, gradient_size, norm_type, border_mode, params.constant_border_value);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CANNY_EDGE_FIXTURE */
