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
#include "NEON/Helper.h"
#include "NEON/NEAccessor.h"
#include "TensorLibrary.h"
#include "benchmark/Datasets.h"
#include "benchmark/Profiler.h"
#include "benchmark/WallClockTimer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include "benchmark/benchmark_api.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::neon;

#include "benchmark/common/PoolingLayer.h"

namespace
{
using PoolingLayerAlexNetF32 = PoolingLayer<AlexNetPoolingLayerDataset, Tensor, NEAccessor, NEPoolingLayer>;
using PoolingLayerAlexNetQS8 = PoolingLayer<AlexNetPoolingLayerDataset, Tensor, NEAccessor, NEPoolingLayer, DataType::QS8>;
using PoolingLayerLeNet5     = PoolingLayer<LeNet5PoolingLayerDataset, Tensor, NEAccessor, NEPoolingLayer>;
using PoolingLayerGoogLeNet  = PoolingLayer<GoogLeNetPoolingLayerDataset, Tensor, NEAccessor, NEPoolingLayer>;
} // namespace

// F32
BENCHMARK_DEFINE_F(PoolingLayerAlexNetF32, neon_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        pool_layer.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(PoolingLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerAlexNetF32, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 2, 1, 4, 8>);

// QS8
BENCHMARK_DEFINE_F(PoolingLayerAlexNetQS8, neon_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        pool_layer.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(PoolingLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerAlexNetQS8, neon_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 2, 1, 4, 8>);

BENCHMARK_DEFINE_F(PoolingLayerLeNet5, neon_lenet5)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        pool_layer.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(PoolingLayerLeNet5, neon_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5PoolingLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerLeNet5, neon_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5PoolingLayerDataset, 1, 1, 4, 8>);

BENCHMARK_DEFINE_F(PoolingLayerGoogLeNet, neon_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        pool_layer.run();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 2, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 3, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 4, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 5, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 6, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 7, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 8, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, neon_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 9, 1, 4, 8>);
