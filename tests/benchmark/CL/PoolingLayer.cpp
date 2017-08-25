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
#include "CL/CLAccessor.h"
#include "CL/Helper.h"
#include "Globals.h"
#include "TensorLibrary.h"
#include "benchmark/Datasets.h"
#include "benchmark/Profiler.h"
#include "benchmark/WallClockTimer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"

#include "benchmark/benchmark.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::cl;

#include "benchmark/common/PoolingLayer.h"

namespace
{
using PoolingLayerAlexNet   = PoolingLayer<AlexNetPoolingLayerDataset, CLTensor, CLAccessor, CLPoolingLayer>;
using PoolingLayerLeNet5    = PoolingLayer<LeNet5PoolingLayerDataset, CLTensor, CLAccessor, CLPoolingLayer>;
using PoolingLayerGoogLeNet = PoolingLayer<GoogLeNetPoolingLayerDataset, CLTensor, CLAccessor, CLPoolingLayer>;
} // namespace

BENCHMARK_DEFINE_F(PoolingLayerAlexNet, cl_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        pool_layer.run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(PoolingLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetPoolingLayerDataset, 2, 1, 4, 8>);

BENCHMARK_DEFINE_F(PoolingLayerLeNet5, cl_lenet5)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        pool_layer.run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(PoolingLayerLeNet5, cl_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5PoolingLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerLeNet5, cl_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5PoolingLayerDataset, 1, 1, 4, 8>);

BENCHMARK_DEFINE_F(PoolingLayerGoogLeNet, cl_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        pool_layer.run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 2, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 3, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 4, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 5, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 6, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 7, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 8, 1, 4, 8>);
BENCHMARK_REGISTER_F(PoolingLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetPoolingLayerDataset, 9, 1, 4, 8>);
