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
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"

#include "benchmark/benchmark_api.h"

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::benchmark;
using namespace arm_compute::test::cl;

#include "benchmark/common/ConvolutionLayer.h"

namespace
{
using ConvolutionLayerAlexNet    = ConvolutionLayer<AlexNetConvolutionLayerDataset, CLTensor, CLAccessor, CLConvolutionLayer>;
using ConvolutionLayerLeNet5     = ConvolutionLayer<LeNet5ConvolutionLayerDataset, CLTensor, CLAccessor, CLConvolutionLayer>;
using ConvolutionLayerGoogLeNet1 = ConvolutionLayer<GoogLeNetConvolutionLayerDataset1, CLTensor, CLAccessor, CLConvolutionLayer>;
using ConvolutionLayerGoogLeNet2 = ConvolutionLayer<GoogLeNetConvolutionLayerDataset2, CLTensor, CLAccessor, CLConvolutionLayer>;
} // namespace

BENCHMARK_DEFINE_F(ConvolutionLayerAlexNet, cl_alexnet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        conv_layer->run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(ConvolutionLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetConvolutionLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetConvolutionLayerDataset, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetConvolutionLayerDataset, 2, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetConvolutionLayerDataset, 3, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerAlexNet, cl_alexnet)
->Threads(1)
->Apply(DataSetArgBatched<AlexNetConvolutionLayerDataset, 4, 1, 4, 8>);

BENCHMARK_DEFINE_F(ConvolutionLayerLeNet5, cl_lenet5)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        conv_layer->run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(ConvolutionLayerLeNet5, cl_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5ConvolutionLayerDataset, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerLeNet5, cl_lenet5)
->Threads(1)
->Apply(DataSetArgBatched<LeNet5ConvolutionLayerDataset, 1, 1, 4, 8>);

BENCHMARK_DEFINE_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        conv_layer->run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_DEFINE_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
(::benchmark::State &state)
{
    while(state.KeepRunning())
    {
        // Run function
        profiler.start();
        conv_layer->run();
        CLScheduler::get().sync();
        profiler.stop();
    }
}

BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 2, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 3, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 4, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 5, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 6, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 7, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 8, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 9, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 10, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 11, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 12, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 13, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 14, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 15, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 16, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 17, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 18, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 19, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 20, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 21, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 22, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 23, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 24, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 25, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 26, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 27, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 28, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 29, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 30, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet1, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset1, 31, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 0, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 1, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 2, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 3, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 4, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 5, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 6, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 7, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 8, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 9, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 10, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 11, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 12, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 13, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 14, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 15, 1, 4, 8>);
BENCHMARK_REGISTER_F(ConvolutionLayerGoogLeNet2, cl_googlenet)
->Threads(1)
->Apply(DataSetArgBatched<GoogLeNetConvolutionLayerDataset2, 16, 1, 4, 8>);
