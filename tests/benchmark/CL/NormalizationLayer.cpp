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
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"

#include "benchmark/benchmark.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::cl;

#include "benchmark/common/NormalizationLayer.h"

namespace
{
using NormalizationLayerAlexNet   = NormalizationLayer<AlexNetNormalizationLayerDataset, CLTensor, CLAccessor, CLNormalizationLayer>;
using NormalizationLayerGoogLeNet = NormalizationLayer<GoogLeNetNormalizationLayerDataset, CLTensor, CLAccessor, CLNormalizationLayer>;

} // namespace

BENCHMARK_DEFINE_F(NormalizationLayerAlexNet, cl_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        norm_layer->run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(NormalizationLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetNormalizationLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(NormalizationLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetNormalizationLayerDataset, 1, 1, 4, 8>);

BENCHMARK_DEFINE_F(NormalizationLayerGoogLeNet, cl_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        norm_layer->run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(NormalizationLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetNormalizationLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(NormalizationLayerGoogLeNet, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetNormalizationLayerDataset, 1, 1, 4, 8>);
