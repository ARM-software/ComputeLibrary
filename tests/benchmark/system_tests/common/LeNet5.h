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
#ifndef __ARM_COMPUTE_TEST_BENCHMARK_LENET5_H__
#define __ARM_COMPUTE_TEST_BENCHMARK_LENET5_H__

#include "tests/TensorLibrary.h"
#include "tests/Utils.h"

#include "benchmark/Profiler.h"
#include "benchmark/WallClockTimer.h"

#include "model_objects/LeNet5.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType,
          typename Accessor,
          typename ActivationLayerFunction,
          typename ConvolutionLayerFunction,
          typename FullyConnectedLayerFunction,
          typename PoolingLayerFunction,
          typename SoftmaxLayerFunction>
class LeNet5Fixture : public ::benchmark::Fixture
{
public:
    void SetUp(::benchmark::State &state) override
    {
        profiler.add(std::make_shared<WallClockTimer>());

        network.build(static_cast<unsigned int>(state.range(0)));
        network.fill_random();
    }

    void TearDown(::benchmark::State &state) override
    {
        profiler.submit(state);
        network.clear();
    }

    Profiler profiler{};
    model_objects::LeNet5<TensorType,
                  Accessor,
                  ActivationLayerFunction,
                  ConvolutionLayerFunction,
                  FullyConnectedLayerFunction,
                  PoolingLayerFunction,
                  SoftmaxLayerFunction>
                  network{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif
