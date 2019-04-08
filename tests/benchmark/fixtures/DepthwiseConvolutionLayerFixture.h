/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_DEPTHWISECONVOLUTIONFIXTURE
#define ARM_COMPUTE_TEST_DEPTHWISECONVOLUTIONFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
using namespace arm_compute::misc::shape_calculator;

/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class DepthwiseConvolutionLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape src_shape, Size2D kernel_size, PadStrideInfo info, Size2D Dilation, DataType data_type, int batches)
    {
        // Get shapes
        TensorShape weights_shape(kernel_size.width, kernel_size.height);

        const TensorInfo in_info(src_shape, 1, data_type);
        const TensorInfo we_info(weights_shape, 1, data_type);
        TensorShape      dst_shape = compute_depthwise_convolution_shape(in_info, we_info, info, 1);

        weights_shape.set(2, dst_shape.z());

        // Set batched in source and destination shapes

        src_shape.set(3 /* batch */, batches);
        dst_shape.set(3 /* batch */, batches);

        // Create tensors
        src     = create_tensor<TensorType>(src_shape, data_type, 1, QuantizationInfo(0.5f, 10));
        weights = create_tensor<TensorType>(weights_shape, data_type, 1, QuantizationInfo(0.5f, 10));
        biases  = create_tensor<TensorType>(TensorShape(weights_shape[2]), is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type, 1);
        dst     = create_tensor<TensorType>(dst_shape, data_type, 1, QuantizationInfo(0.5f, 10));

        // Create and configure function
        depth_conv.configure(&src, &weights, &biases, &dst, info);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        biases.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        depth_conv.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        src.allocator()->free();
        weights.allocator()->free();
        biases.allocator()->free();
        dst.allocator()->free();
    }

private:
    TensorType src{};
    TensorType weights{};
    TensorType biases{};
    TensorType dst{};
    Function   depth_conv{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISECONVOLUTIONFIXTURE */
