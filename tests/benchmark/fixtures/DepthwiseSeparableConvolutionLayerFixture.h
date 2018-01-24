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
#ifndef ARM_COMPUTE_TEST_DEPTHWISESEPARABLECONVOLUTIONLAYERFIXTURE
#define ARM_COMPUTE_TEST_DEPTHWISESEPARABLECONVOLUTIONLAYERFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class DepthwiseSeparableConvolutionLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape src_shape, TensorShape depthwise_weights_shape, TensorShape depthwise_biases_shape, TensorShape depthwise_out_shape, TensorShape pointwise_weights_shape,
               TensorShape pointwise_biases_shape, TensorShape dst_shape,
               PadStrideInfo pad_stride_depthwise_info, PadStrideInfo pad_stride_pointwise_info, DataType data_type, int batches)
    {
        // Set batched in source and destination shapes
        const unsigned int fixed_point_position = 4;
        src_shape.set(3 /* batch */, batches);
        depthwise_out_shape.set(3 /* batch */, batches);
        dst_shape.set(3 /* batch */, batches);

        src               = create_tensor<TensorType>(src_shape, data_type, 1, fixed_point_position);
        depthwise_weights = create_tensor<TensorType>(depthwise_weights_shape, data_type, 1, fixed_point_position);
        depthwise_biases  = create_tensor<TensorType>(depthwise_biases_shape, data_type, 1, fixed_point_position);
        depthwise_out     = create_tensor<TensorType>(depthwise_out_shape, data_type, 1, fixed_point_position);
        pointwise_weights = create_tensor<TensorType>(pointwise_weights_shape, data_type, 1, fixed_point_position);
        pointwise_biases  = create_tensor<TensorType>(pointwise_biases_shape, data_type, 1, fixed_point_position);
        dst               = create_tensor<TensorType>(dst_shape, data_type, 1, fixed_point_position);

        // Create and configure function
        depth_sep_conv_layer.configure(&src, &depthwise_weights, &depthwise_biases, &depthwise_out, &pointwise_weights, &pointwise_biases, &dst, pad_stride_depthwise_info, pad_stride_pointwise_info);

        // Allocate tensors
        src.allocator()->allocate();
        depthwise_weights.allocator()->allocate();
        depthwise_biases.allocator()->allocate();
        depthwise_out.allocator()->allocate();
        pointwise_weights.allocator()->allocate();
        pointwise_biases.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        depth_sep_conv_layer.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        src.allocator()->free();
        depthwise_weights.allocator()->free();
        depthwise_biases.allocator()->free();
        depthwise_out.allocator()->free();
        pointwise_weights.allocator()->free();
        pointwise_biases.allocator()->free();
        dst.allocator()->free();
    }

private:
    TensorType src{};
    TensorType depthwise_weights{};
    TensorType depthwise_biases{};
    TensorType depthwise_out{};
    TensorType pointwise_weights{};
    TensorType pointwise_biases{};
    TensorType dst{};
    Function   depth_sep_conv_layer{};
};
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISESEPARABLECONVOLUTIONLAYERFIXTURE */
