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
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "benchmark/benchmark_api.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::neon;

#include "benchmark/common/ActivationLayer.h"

namespace
{
using ActivationLayerAlexNetF32 = ActivationLayer<AlexNetActivationLayerDataset, Tensor, NEAccessor, NEActivationLayer>;
using ActivationLayerAlexNetQS8 = ActivationLayer<AlexNetActivationLayerDataset, Tensor, NEAccessor, NEActivationLayer, DataType::QS8>;
using ActivationLayerLeNet5     = ActivationLayer<LeNet5ActivationLayerDataset, Tensor, NEAccessor, NEActivationLayer, DataType::F32>;
using ActivationLayerGoogLeNet  = ActivationLayer<GoogLeNetActivationLayerDataset, Tensor, NEAccessor, NEActivationLayer, DataType::F32>;
} // namespace

// F32
BENCHMARK_DEFINE_F(ActivationLayerAlexNetF32, neon_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        act_layer.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(ActivationLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 2, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 3, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 4, 1, 4, 8>);

// QS8
BENCHMARK_DEFINE_F(ActivationLayerAlexNetQS8, neon_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        act_layer.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(ActivationLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 2, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 3, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetActivationLayerDataset, 4, 1, 4, 8>);

BENCHMARK_DEFINE_F(ActivationLayerLeNet5, neon_lenet5)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        act_layer.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(ActivationLayerLeNet5, neon_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5ActivationLayerDataset, 0, 1, 4, 8>);

BENCHMARK_DEFINE_F(ActivationLayerGoogLeNet, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        act_layer.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 2, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 3, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 4, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 5, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 6, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 7, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 8, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 9, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 10, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 11, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 12, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 13, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 14, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 15, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 16, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 17, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 18, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 19, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 20, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 21, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 22, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 23, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 24, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 25, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 26, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 27, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 28, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 29, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 30, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 31, 1, 4, 8>);
BENCHMARK_REGISTER_F(ActivationLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetActivationLayerDataset, 32, 1, 4, 8>);
