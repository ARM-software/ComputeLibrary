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
//FIXME / INTERNAL_ONLY: This file should not be released!
#ifndef ARM_COMPUTE_TEST_DRAGONBENCH_FIXTURE
#define ARM_COMPUTE_TEST_DRAGONBENCH_FIXTURE

#include "arm_compute/core/Error.h"
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
template <typename TensorType, typename Function, typename Accessor, typename Conv2DConfig>
class DragonBenchConv2DFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(Conv2DConfig config, DataType data_type, DataLayout data_layout, bool has_bias)
    {
        // Set tensor shapes in NCHW layout
        TensorShape src_shape(config.dim_in_w, config.dim_in_h, config.ch_in, config.ibatch);
        TensorShape weights_shape(config.kern_w, config.kern_h, config.ch_in, config.ch_out);
        TensorShape biases_shape(config.ch_out);
        TensorShape dst_shape(config.dim_out_w, config.dim_out_h, config.ch_out, config.ibatch);

        // Set convolution layer info
        PadStrideInfo info(config.stride_w, config.stride_h, 0, 0);
        if(config.padding)
        {
            info = calculate_same_pad(src_shape, weights_shape, info);
        }

        // Permute shapes in case of NHWC
        if(data_layout == DataLayout::NHWC)
        {
            permute(src_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(dst_shape, PermutationVector(2U, 0U, 1U));
        }

        // Determine bias data type
        DataType bias_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

        // Create tensors
        src     = create_tensor<TensorType>(src_shape, data_type, 1, QuantizationInfo(), data_layout);
        weights = create_tensor<TensorType>(weights_shape, data_type, 1, QuantizationInfo(), data_layout);
        biases  = create_tensor<TensorType>(biases_shape, bias_data_type, 1, QuantizationInfo(), data_layout);
        dst     = create_tensor<TensorType>(dst_shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        conv_layer.configure(&src, &weights, has_bias ? &biases : nullptr, &dst, info);

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
#endif /* ARM_COMPUTE_TEST_DRAGONBENCH_FIXTURE */
