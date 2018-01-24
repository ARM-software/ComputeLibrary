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
#ifndef ARM_COMPUTE_TEST_ALEXNETFIXTURE
#define ARM_COMPUTE_TEST_ALEXNETFIXTURE

#include "tests/AssetsLibrary.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"
#include "tests/networks/AlexNetNetwork.h"

namespace arm_compute
{
namespace test
{
template <typename ITensorType,
          typename TensorType,
          typename SubTensorType,
          typename Accessor,
          typename ActivationLayerFunction,
          typename ConvolutionLayerFunction,
          typename DirectConvolutionLayerFunction,
          typename FullyConnectedLayerFunction,
          typename NormalizationLayerFunction,
          typename PoolingLayerFunction,
          typename SoftmaxLayerFunction>

class AlexNetFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(DataType data_type, int batches)
    {
        constexpr bool weights_reshaped     = false;
        constexpr int  fixed_point_position = 4;

        network.init(data_type, fixed_point_position, batches, weights_reshaped);
        network.build();
        network.allocate();
    }

    void run()
    {
        network.run();
    }

    void sync()
    {
        network.sync();
    }

    void teardown()
    {
        network.clear();
    }

private:
    networks::AlexNetNetwork<ITensorType,
             TensorType,
             SubTensorType,
             Accessor,
             ActivationLayerFunction,
             ConvolutionLayerFunction,
             DirectConvolutionLayerFunction,
             FullyConnectedLayerFunction,
             NormalizationLayerFunction,
             PoolingLayerFunction,
             SoftmaxLayerFunction>
             network{};
};
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ALEXNETFIXTURE */
