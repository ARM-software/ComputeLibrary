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
#include "Globals.h"
#include "NEON/NEAccessor.h"
#include "TensorLibrary.h"
#include "benchmark/Datasets.h"
#include "benchmark/Profiler.h"
#include "benchmark/WallClockTimer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "benchmark/benchmark_api.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::neon;

#include "benchmark/common/FullyConnectedLayer.h"

namespace
{
using FullyConnectedLayerAlexNetF32 = FullyConnectedLayer<AlexNetFullyConnectedLayerDataset, Tensor, NEAccessor, NEFullyConnectedLayer>;
using FullyConnectedLayerAlexNetQS8 = FullyConnectedLayer<AlexNetFullyConnectedLayerDataset, Tensor, NEAccessor, NEFullyConnectedLayer, DataType::QS8>;
using FullyConnectedLayerLeNet5     = FullyConnectedLayer<LeNet5FullyConnectedLayerDataset, Tensor, NEAccessor, NEFullyConnectedLayer>;
using FullyConnectedLayerGoogLeNet  = FullyConnectedLayer<GoogLeNetFullyConnectedLayerDataset, Tensor, NEAccessor, NEFullyConnectedLayer>;
} // namespace

// F32
BENCHMARK_DEFINE_F(FullyConnectedLayerAlexNetF32, neon_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        fc_layer->run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(FullyConnectedLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetFullyConnectedLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(FullyConnectedLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetFullyConnectedLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(FullyConnectedLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetFullyConnectedLayerDataset, 2, 1, 4, 8>);

// QS8
BENCHMARK_DEFINE_F(FullyConnectedLayerAlexNetQS8, neon_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        fc_layer->run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(FullyConnectedLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetFullyConnectedLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(FullyConnectedLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetFullyConnectedLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(FullyConnectedLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetFullyConnectedLayerDataset, 2, 1, 4, 8>);

BENCHMARK_DEFINE_F(FullyConnectedLayerLeNet5, neon_lenet5)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        fc_layer->run();
        profiler.stop();
    }
}
BENCHMARK_REGISTER_F(FullyConnectedLayerLeNet5, neon_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5FullyConnectedLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(FullyConnectedLayerLeNet5, neon_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5FullyConnectedLayerDataset, 1, 1, 4, 8>);

BENCHMARK_DEFINE_F(FullyConnectedLayerGoogLeNet, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        fc_layer->run();
        profiler.stop();
    }
}
BENCHMARK_REGISTER_F(FullyConnectedLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetFullyConnectedLayerDataset, 0, 1, 4, 8>);
