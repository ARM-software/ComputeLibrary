/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_FFT_CONVOLUTION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_FFT_CONVOLUTION_LAYER_FIXTURE

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
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class FFTConvolutionLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape src_shape, TensorShape weights_shape, TensorShape biases_shape, TensorShape dst_shape, PadStrideInfo info, Size2D dilation, ActivationLayerInfo act_info, DataType data_type,
               int batches)
    {
        ARM_COMPUTE_UNUSED(dilation);

        // Set batched in source and destination shapes

        src_shape.set(3 /* batch */, batches);
        dst_shape.set(3 /* batch */, batches);

        // Create tensors
        src     = create_tensor<TensorType>(src_shape, data_type, 1);
        weights = create_tensor<TensorType>(weights_shape, data_type, 1);
        biases  = create_tensor<TensorType>(biases_shape, data_type, 1);
        dst     = create_tensor<TensorType>(dst_shape, data_type, 1);

        // Create and configure function
        conv_layer.configure(&src, &weights, &biases, &dst, info, act_info);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        biases.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        conv_layer.run();
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
    Function   conv_layer{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FFT_CONVOLUTION_LAYER_FIXTURE */
