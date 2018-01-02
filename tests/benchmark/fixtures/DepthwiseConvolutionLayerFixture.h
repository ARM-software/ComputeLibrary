/*
 * Copyright (c) 2017 ARM Limited.
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
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class DepthwiseConvolutionLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape src_shape, TensorShape weights_shape, TensorShape biases_shape, TensorShape dst_shape, PadStrideInfo info, DataType data_type, int batches)
    {
        // Set batched in source and destination shapes
        const unsigned int fixed_point_position = 4;
        src_shape.set(3 /* batch */, batches);
        dst_shape.set(3 /* batch */, batches);

        // Create tensors
        src     = create_tensor<TensorType>(src_shape, data_type, 1, fixed_point_position);
        weights = create_tensor<TensorType>(weights_shape, data_type, 1, fixed_point_position);
        biases  = create_tensor<TensorType>(biases_shape, data_type, 1, fixed_point_position);
        dst     = create_tensor<TensorType>(dst_shape, data_type, 1, fixed_point_position);

        // Create and configure function
        depth_conv.configure(&src, &weights, &biases, &dst, info);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        biases.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill tensors
        library->fill_tensor_uniform(Accessor(src), 0);
        library->fill_tensor_uniform(Accessor(weights), 1);
    }

    void run()
    {
        depth_conv.run();
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
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISECONVOLUTIONFIXTURE */
